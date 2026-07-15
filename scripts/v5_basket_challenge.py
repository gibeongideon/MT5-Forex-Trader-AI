"""Diversified BASKET book engine for the FundingPips challenge.

Same champion long-only recipe as the single-XAU book, applied across every
tradeable structural-drift class the FundingPips account offers, combined
class-equal-risk. Diversification shrinks intraday excursions vs the 5% daily
line -> the realistic pass rate rises from ~61% (single-XAU) to ~80%, daily-loss
blowups from ~32% to ~7% (see CHALLENGEBOT.MD / fp_sim).

Two entry points:
  --backtest   reconstruct the book, print Sharpe + FundingPips pass sim
               (validates this engine reproduces the lab's ~1.26 SR / 80% pass)
  --targets    print the LIVE per-symbol target leverage as of the latest bar
               (fixed per-symbol weight W_i x vol-targeted position — what the
               executor reconciles to; broker order-wiring is added when the
               FundingPips account/terminal exists)

Book construction (all causal, next-bar applied, net of the D1 spread column):
  per asset  pos_i = champ_recipe(close) * (10%/vol_i), buffered   (~10% vol each)
  weight     W_i   = k * g/Nclass * b_cls / Nmembers * a_i
             where a_i, b_cls, g are the z-to-10% scalars at asset/class/portfolio
             level (constants from history) -> account book ~= k*10% vol.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/home/rock/Desktop/2026_Projects/Trader36/MT5"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data/v5_runs/xau-sharpe1-lab"))
sys.path.insert(0, os.path.join(ROOT, "data/v5_runs/challenge-lab"))

from xau_lab import ewmac_fc, breakout_fc, load_h4, ANN_H4, SLIP_USD  # noqa
from challenge_lab import fp_sim  # noqa

# tradeable classes -> members (FundingPips: indices + crypto + metals; NO rates/energy)
CLASSES = {
    "eq_us": ["SPX", "NDX", "DJI"],
    "eq_eu": ["DAX", "FTSE", "STOXX"],
    "eq_ap": ["NIKKEI", "ASX"],
    "crypto": ["BTC", "ETH"],
    "xau": ["XAUCHAMP"],       # H4 champion (special-cased loader)
    "metal": ["SILVER"],
}
TARGET_VOL = 0.10
BUF = 0.1

# FundingPips model presets. `vol` = chosen account vol dial (validated safest
# operating point per model, CHALLENGEBOT.MD / fp_sim). guard/halt/targets feed
# the executor's challenge_guards. STANDARD @ 7% is the locked default: ~92%
# pass, 0% daily-loss breaches (5% daily limit is never approached).
MODELS = {
    "standard": dict(vol=0.07, p1=0.08, p2=0.05, daily=0.05, maxloss=0.10,
                     guard_frac=0.035, halt_frac=0.08),
    "flex":     dict(vol=0.07, p1=0.10, p2=0.06, daily=0.04, maxloss=0.12,
                     guard_frac=0.030, halt_frac=0.10),
    "pro":      dict(vol=0.05, p1=0.06, p2=0.06, daily=0.03, maxloss=0.06,
                     guard_frac=0.022, halt_frac=0.05),
    "onestep":  dict(vol=0.07, p1=0.10, p2=0.00, daily=0.05, maxloss=0.06,
                     guard_frac=0.035, halt_frac=0.05),
}
DEFAULT_MODEL = "standard"
K_DIAL = MODELS[DEFAULT_MODEL]["vol"] / TARGET_VOL   # dial so book ~= model vol


def _norm(s):
    return s * (1.0 / s.abs().expanding(min_periods=120).mean().shift(1))


def _conc(s, p=1.5):
    return _norm(s.clip(lower=0.0) ** p)


def champ_recipe_lo(close):
    ew = ewmac_fc(close, ((16, 64), (32, 128), (64, 256)))
    bk = breakout_fc(close, (10, 20, 40))
    return (0.5 * (_conc(np.maximum(ew.clip(lower=0), bk.clip(lower=0))) * 0.8 + 0.15)
            + 0.5 * (_conc(bk) * 0.8 + 0.15)).clip(0, 2)


def _buffered_pos(fc, vol, spread_px, close, ann):
    pos = (fc * (TARGET_VOL / vol)).clip(-8, 8)
    band = BUF * (TARGET_VOL / vol).clip(0, 8)
    p, out, held = pos.values, np.zeros(len(pos)), 0.0
    for i in range(len(p)):
        if np.isfinite(p[i]):
            b = band.iloc[i] if np.isfinite(band.iloc[i]) else 0.0
            if abs(p[i] - held) > b:
                held = p[i] - np.sign(p[i] - held) * b
        out[i] = held
    return pd.Series(out, index=pos.index)


def _load_asset(sym):
    """Return (position_series[10%vol], net_daily_stream) for one asset."""
    if sym == "XAUCHAMP":
        df = load_h4()
        D = 6
        ew = ewmac_fc(df["close"], tuple((f * D, s * D) for f, s in ((16, 64), (32, 128), (64, 256))))
        bk = breakout_fc(df["close"], [d * D for d in (10, 20, 40)])
        fc = (0.5 * (_conc(np.maximum(ew.clip(lower=0), bk.clip(lower=0))) * 0.8 + 0.15)
              + 0.5 * (_conc(bk) * 0.8 + 0.15)).clip(0, 2)
        ann, cost_px = ANN_H4, (df["spread_px"] / 2 + SLIP_USD)
    else:
        df = pd.read_csv(f"{ROOT}/data/{sym}_D1_long.csv", parse_dates=["time"], index_col="time").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df["spread_px"] = df["spread"].clip(lower=df["spread"].median())
        fc = champ_recipe_lo(df["close"])
        ann, cost_px = 252, df["spread_px"]
    close = df["close"]
    ret = close.pct_change()
    vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(ann)
    pos = _buffered_pos(fc, vol, df["spread_px"], close, ann).shift(1).fillna(0.0)
    cost = pos.diff().abs().fillna(0.0) * (cost_px / close)
    net = (pos * ret - cost).fillna(0.0).resample("D").sum()
    net = net[net.index.dayofweek < 5]
    # live target leverage (pre-weight): the vol-targeted position itself, latest value
    live_pos = _buffered_pos(fc, vol, df["spread_px"], close, ann)
    return live_pos, net


def build(start="2016-01-01", dial=K_DIAL):
    """Return (weights dict W_i, account_book daily series, per-asset live pos)."""
    live, nets = {}, {}
    for members in CLASSES.values():
        for sym in members:
            lp, nt = _load_asset(sym)
            live[sym], nets[sym] = lp, nt.loc[start:]
    al = pd.DataFrame(nets).loc[start:]

    def z(d):
        sd = d.std() * np.sqrt(252)
        return d * (TARGET_VOL / sd) if sd > 0 else d

    a = {sym: TARGET_VOL / (al[sym].std() * np.sqrt(252)) for sym in al.columns}   # asset scalar
    cls_stream, b_cls = {}, {}
    for cls, members in CLASSES.items():
        members = [m for m in members if m in al.columns]
        comp = sum(a[m] * al[m].fillna(0.0) for m in members) / len(members)
        b_cls[cls] = TARGET_VOL / (comp.std() * np.sqrt(252))
        cls_stream[cls] = b_cls[cls] * comp
    cl = pd.DataFrame(cls_stream).dropna()
    port = sum(cl[c] for c in cl.columns) / len(cl.columns)
    g = TARGET_VOL / (port.std() * np.sqrt(252))
    book = dial * g * port

    W = {}
    Nc = len(CLASSES)
    for cls, members in CLASSES.items():
        members = [m for m in members if m in al.columns]
        for sym in members:
            W[sym] = dial * g / Nc * b_cls[cls] / len(members) * a[sym]
    return W, book, live


def target_leverage(model=DEFAULT_MODEL):
    """Public API for the executor: {symbol: target account-leverage} at the
    latest bar, sized for the given model's vol dial. Long-only (>=0)."""
    dial = MODELS[model]["vol"] / TARGET_VOL
    W, _, live = build(dial=dial)
    out = {}
    for sym in W:
        s = live[sym].dropna()
        p = float(s.iloc[-1]) if len(s) else 0.0
        out[sym] = max(0.0, W[sym] * p)     # champion recipe is long-only
    return out


def cmd_backtest(model=DEFAULT_MODEL):
    m = MODELS[model]
    dial = m["vol"] / TARGET_VOL
    W, book, _ = build(dial=dial)
    sh = lambda d, s="2017-01-01": float(d.loc[s:].mean() / d.loc[s:].std() * np.sqrt(252))
    eq = (1 + book.loc["2017-01-01":]).cumprod()
    print(f"=== BASKET book (this engine)  model={model.upper()} vol~{m['vol']*100:.0f}% ===")
    print(f"  eval SR {sh(book):+.3f}   full {sh(book,'2016-06-01'):+.3f}   "
          f"2021+ {sh(book,'2021-01-01'):+.3f}   vol {book.std()*np.sqrt(252)*100:.1f}%   "
          f"maxDD {float((eq/eq.cummax()-1).min()*100):.1f}%")
    print("  per-symbol weight W_i (live leverage multiplier):")
    for sym, w in sorted(W.items(), key=lambda x: -x[1]):
        print(f"    {sym:9s} {w:+.3f}")
    print(f"\n=== FundingPips {model.upper()} pass sim (P1 {m['p1']*100:.0f}%/P2 {m['p2']*100:.0f}%, "
          f"daily {m['daily']*100:.0f}%, max {m['maxloss']*100:.0f}%) ===")
    r10 = (book * (0.10 / (book.std() * np.sqrt(252)))).values
    for tag, ds in (("idealized", 1.0), ("realistic ", 1.5)):
        s = fp_sim(r10, dial, day_safety=ds, p1=m["p1"], p2=m["p2"],
                   dayloss=m["daily"], maxloss=m["maxloss"])
        print(f"  {tag}  vol~{m['vol']*100:.0f}%  pass {s['passpct']:5.1f}%  "
              f"failDay {s['fail_day']:4.1f}%  failDD {s['fail_dd']:4.1f}%  "
              f"median {s['med_mo']:.1f}mo  p75 {s['q75_mo']:.1f}mo")


def cmd_targets(model=DEFAULT_MODEL):
    dial = MODELS[model]["vol"] / TARGET_VOL
    W, _, live = build(dial=dial)
    print("=== LIVE per-symbol target leverage (latest bar) ===")
    print(f"{'symbol':9s} {'W_i':>7s} {'pos(vol-tgt)':>13s} {'TARGET lev':>11s}")
    tot = 0.0
    for sym in sorted(W):
        p = float(live[sym].dropna().iloc[-1]) if len(live[sym].dropna()) else 0.0
        tgt = W[sym] * p
        tot += abs(tgt)
        print(f"{sym:9s} {W[sym]:+7.3f} {p:+13.2f} {tgt:+11.3f}")
    print(f"{'TOTAL gross':9s} {'':7s} {'':13s} {tot:+11.3f}")
    print("\nNOTE: convert TARGET lev -> lots at execution time using account equity,\n"
          "per-symbol contract size, and the FundingPips symbol names (deferred).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--backtest", action="store_true")
    g.add_argument("--targets", action="store_true")
    args = ap.parse_args()
    if args.backtest:
        cmd_backtest(args.model)
    else:
        cmd_targets(args.model)


if __name__ == "__main__":
    main()
