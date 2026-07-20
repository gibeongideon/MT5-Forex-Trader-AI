"""What happens to the trend book if weekend / overnight holding is BANNED?

Funded (Master) accounts restrict holding: FundingPips banned weekend holding on
2-Step Flex Masters (29-Jan-2026, auto-closed Friday); some firms also forbid
overnight. This quantifies the damage instead of guessing.

Scenarios (same champion long-only recipe + vol targeting each time):
  BASELINE      ret = close/prev_close - 1        (hold through everything)
  WEEKEND-FLAT  flat Fri close -> re-enter Mon open: Monday earns only
                close/open-1, so the Fri->Mon GAP is missed; +2 crossings/week
  OVERNIGHT-FLAT flat every night: every day earns only close/open-1, so EVERY
                close->open gap is missed; +2 crossings/day

The gap is exactly where trend-following earns much of its money, so this is the
honest test of whether the strategy survives the funded-stage rules.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

ROOT = "/home/rock/Desktop/2026_Projects/Trader36/MT5"
sys.path.insert(0, ROOT + "/scripts")
import v5_basket_challenge as vbc  # noqa

EVAL = "2017-01-01"
TARGET_VOL = 0.10
MODEL = vbc.MODELS["flex"]
DIAL = MODEL["vol"] / TARGET_VOL


def load_d1(sym):
    df = pd.read_csv(f"{ROOT}/data/{sym}_D1_long.csv", parse_dates=["time"],
                     index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["spread_px"] = df["spread"].clip(lower=df["spread"].median())
    return df


def sleeve(sym, mode):
    """Return daily net return stream for one sleeve under a holding constraint."""
    df = load_d1(sym)
    close, opn = df["close"], df["open"]
    fc = vbc.champ_recipe_lo(close)                 # long-only champion forecast
    ret_full = close.pct_change()                   # hold-through return
    ret_intra = (close / opn) - 1.0                 # open->close only (no gap)

    is_mon = df.index.dayofweek == 0
    if mode == "baseline":
        ret, extra_turns = ret_full, 0.0
    elif mode == "weekend":
        ret = ret_full.where(~is_mon, ret_intra)    # miss only the Fri->Mon gap
        extra_turns = 2.0 / 5.0                     # ~2 crossings per 5 trading days
    elif mode == "overnight":
        ret = ret_intra                             # miss EVERY overnight gap
        extra_turns = 2.0                           # 2 crossings every day
    else:
        raise ValueError(mode)

    vol = ret_full.ewm(halflife=42, min_periods=20).std() * np.sqrt(252)
    pos = (fc * (TARGET_VOL / vol)).clip(0, 8).shift(1).fillna(0.0)
    cost_frac = df["spread_px"] / close
    # normal rebalancing cost + forced flat/re-entry cost
    cost = pos.diff().abs().fillna(0.0) * cost_frac + pos.abs() * extra_turns * cost_frac
    return (pos * ret - cost).fillna(0.0)


def book(syms, mode):
    streams = {s: sleeve(s, mode).loc[EVAL:] for s in syms}
    al = pd.DataFrame(streams).dropna()
    # equal risk per sleeve, then scale the book to the model vol dial
    z = {s: TARGET_VOL / (al[s].std() * np.sqrt(252)) for s in al.columns}
    port = sum(z[s] * al[s] for s in al.columns) / len(al.columns)
    g = TARGET_VOL / (port.std() * np.sqrt(252))
    b = DIAL * g * port
    # causal portfolio vol-target + drawdown scaler (as live)
    rv = b.ewm(halflife=20, min_periods=20).std() * np.sqrt(252)
    vs = (MODEL["vol"] / rv).clip(0, 3.0)
    eq = (1 + b).cumprod()
    ds = (1 + (eq / eq.cummax() - 1) * 3.0).clip(lower=0.5)
    return (b * (vs * ds).shift(1)).dropna()


def stats(b, label):
    sr = float(b.mean() / b.std() * np.sqrt(252))
    eq = (1 + b).cumprod()
    dd = float((eq / eq.cummax() - 1).min() * 100)
    cagr = float(eq.iloc[-1] ** (252 / len(b)) - 1) * 100
    r10 = (b * (0.10 / (b.std() * np.sqrt(252)))).values
    fp = vbc.fp_sim(r10, DIAL, day_safety=1.5, p1=MODEL["p1"], p2=MODEL["p2"],
                    dayloss=MODEL["daily"], maxloss=MODEL["maxloss"])
    return dict(label=label, sr=sr, dd=dd, cagr=cagr,
                passpct=fp["passpct"], med=fp["med_mo"])


if __name__ == "__main__":
    SYMS = ["GOLD", "ETH", "DJI"]        # FundingPips 10K book (D1 proxies)
    print(f"Book: {' + '.join(SYMS)}   FundingPips FLEX rules, eval {EVAL}+\n")
    print(f"{'scenario':16s} {'Sharpe':>7} {'CAGR%':>7} {'maxDD%':>7} {'pass%':>7} {'median_mo':>10}")
    rows = []
    for mode, label in (("baseline", "hold-through"),
                        ("weekend", "NO weekend"),
                        ("overnight", "NO overnight")):
        s = stats(book(SYMS, mode), label)
        rows.append(s)
        print(f"{s['label']:16s} {s['sr']:+7.2f} {s['cagr']:7.1f} {s['dd']:7.1f} "
              f"{s['passpct']:7.1f} {s['med']:10.1f}")
    base = rows[0]
    print("\nDamage vs hold-through:")
    for s in rows[1:]:
        print(f"  {s['label']:14s} Sharpe {s['sr']/base['sr']*100:5.0f}% of baseline, "
              f"pass {s['passpct']:.1f}% (was {base['passpct']:.1f}%)")
