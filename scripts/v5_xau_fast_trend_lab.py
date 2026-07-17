"""V5 XAUUSD FAST / INTRADAY TREND research lab.

Goal: find a SHORT-TERM trend edge to complement the slow H4 long-only
champion (acct 360542) on the cent account — higher turnover, opens/closes
more trades per day. NOT mean-reversion (fade/reversal are DEAD net of
spread, see V5_FINDINGS.md). Trend/breakout only.

Cost model = the honest cent-account cost: $0.34 round-trip spread. Charged
as half-spread per position change (|dpos| * 0.17/price), so a full round
trip pays the whole $0.34. Optional slippage add-on. Eval window 2017+.
Sharpe from daily-resampled net equity * sqrt(252) (repo convention).

Vectorized continuous vol-targeted CTA PnL for fast sweeping; winners get
re-checked in the discrete lot/stop engine separately. All signals causal
(closes<=t, shift(1) scalars). Correlation vs the champion is reported so we
know whether a candidate DIVERSIFIES the live book or just duplicates it.

    python scripts/v5_xau_fast_trend_lab.py            # full sweep
    python scripts/v5_xau_fast_trend_lab.py --tf H1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EVAL_START = "2017-01-01"
SPREAD_USD = 0.34            # cent-account true round-trip gold spread
HALF_SPREAD = SPREAD_USD / 2.0
TARGET_ANN_VOL = 0.15        # scale every book to 15% ann vol before costs
MAX_LEV = 5.0                # cap notional/equity (realism)
VOL_SPAN_BARS = {"M15": 96, "M30": 48, "H1": 32}   # ~1 day of bars
BARS_PER_YEAR = {"M15": 252 * 96, "M30": 252 * 48, "H1": 252 * 24}

DATA = {"M15": "data/XAUUSD_M15_long.csv",
        "M30": "data/XAUUSD_M30_long.csv",
        "H1": "data/XAUUSD_H1_long.csv"}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load(tf: str) -> pd.DataFrame:
    df = pd.read_csv(DATA[tf], parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[["open", "high", "low", "close"]].astype(float)


# --------------------------------------------------------------------------
# trend forecasts  (continuous, |1| ~ average strength, causal)
# --------------------------------------------------------------------------
def ewmac_fc(close: pd.Series, pairs, cap: float = 2.0) -> pd.Series:
    ret = close.pct_change()
    price_vol = close * ret.ewm(span=36, min_periods=20).std()
    combined = None
    for fast, slow in pairs:
        raw = (close.ewm(span=fast, min_periods=fast).mean()
               - close.ewm(span=slow, min_periods=slow).mean()) / price_vol
        scalar = 1.0 / raw.abs().expanding(min_periods=60).mean().shift(1)
        fc = (raw * scalar).clip(-cap * 2, cap * 2)
        combined = fc if combined is None else combined + fc
    return (combined / len(pairs)).clip(-cap, cap)


def breakout_fc(close: pd.Series, windows, cap: float = 2.0) -> pd.Series:
    combined = None
    for n in windows:
        hi = close.rolling(n, min_periods=n // 2).max()
        lo = close.rolling(n, min_periods=n // 2).min()
        mid = (hi + lo) / 2.0
        rng = (hi - lo).replace(0.0, np.nan)
        raw = ((close - mid) / rng * 4.0).ewm(span=max(2, n // 4)).mean()
        scalar = 1.0 / raw.abs().expanding(min_periods=60).mean().shift(1)
        fc = (raw * scalar).clip(-cap * 2, cap * 2)
        combined = fc if combined is None else combined + fc
    return (combined / len(windows)).clip(-cap, cap)


def orb_fc(df: pd.DataFrame, open_hour: int, range_bars: int,
           close_hour: int) -> pd.Series:
    """Opening-Range Breakout: each session, form range over the first
    `range_bars` bars after open_hour (UTC), then hold +1 if price breaks
    above the range high / -1 below the low, flat after close_hour and
    overnight. Position is a step function (magnitude 1)."""
    h = df.index.hour
    day = df.index.normalize()
    close = df["close"].values
    pos = np.zeros(len(df))
    # per-day opening range
    in_sess = (h >= open_hour) & (h < close_hour)
    rng_hi = pd.Series(df["high"]).where((h >= open_hour) &
                                         (h < open_hour + range_bars // 1))
    # simpler bar-count range: track per group
    fc = np.zeros(len(df))
    df2 = df.assign(_day=day, _h=h)
    for _, g in df2.groupby("_day"):
        gi = g.index
        sess = g[(g["_h"] >= open_hour) & (g["_h"] < close_hour)]
        if len(sess) <= range_bars:
            continue
        opener = sess.iloc[:range_bars]
        rhi, rlo = opener["high"].max(), opener["low"].min()
        rest = sess.iloc[range_bars:]
        state = 0
        for ts, row in rest.iterrows():
            if state == 0:
                if row["close"] > rhi:
                    state = 1
                elif row["close"] < rlo:
                    state = -1
            fc[df.index.get_loc(ts)] = state
    return pd.Series(fc, index=df.index)


def intraday_mom_fc(df: pd.DataFrame, split_hour: int) -> pd.Series:
    """Gao-Han-Zhou intraday momentum: sign of the session's morning return
    (open .. split_hour) sets the position for the rest of that session,
    flat overnight."""
    h = df.index.hour
    day = df.index.normalize()
    fc = np.zeros(len(df))
    df2 = df.assign(_day=day, _h=h)
    for _, g in df2.groupby("_day"):
        if len(g) < 2:
            continue
        sess_open = g["open"].iloc[0]
        morning = g[g["_h"] < split_hour]
        if len(morning) == 0:
            continue
        mid_px = morning["close"].iloc[-1]
        sign = np.sign(mid_px - sess_open)
        aft = g[g["_h"] >= split_hour]
        for ts in aft.index:
            fc[df.index.get_loc(ts)] = sign
    return pd.Series(fc, index=df.index)


# --------------------------------------------------------------------------
# vectorized vol-targeted PnL, net of $0.34 spread
# --------------------------------------------------------------------------
def _buffer(target: np.ndarray, width: np.ndarray) -> np.ndarray:
    """Carver no-trade buffer: only move current position to the nearest
    edge of a band of half-width `width` around the target. Cuts turnover
    hugely with minimal signal loss — the key lever when spread bites."""
    pos = np.zeros(len(target))
    cur = 0.0
    for i in range(len(target)):
        w = width[i] if np.isfinite(width[i]) else 0.0
        lo, hi = target[i] - w, target[i] + w
        if cur < lo:
            cur = lo
        elif cur > hi:
            cur = hi
        pos[i] = cur
    return pos


def backtest(df: pd.DataFrame, fc: pd.Series, tf: str, *, long_only: bool,
             extra_slip_usd: float = 0.0, buffer_frac: float = 0.0) -> dict:
    close = df["close"]
    ret = close.pct_change().fillna(0.0)
    f = fc.reindex(df.index).fillna(0.0)
    if long_only:
        f = f.clip(lower=0.0)

    # scale to target vol using causal trailing return vol
    sig = ret.ewm(span=VOL_SPAN_BARS[tf], min_periods=20).std().shift(1)
    tgt_bar = TARGET_ANN_VOL / np.sqrt(BARS_PER_YEAR[tf])
    lev = (f * tgt_bar / sig.replace(0.0, np.nan)).clip(-MAX_LEV, MAX_LEV)
    lev = lev.fillna(0.0)

    if buffer_frac > 0.0:
        # band half-width = buffer_frac * average target leverage magnitude
        avg_mag = lev.abs().ewm(span=BARS_PER_YEAR[tf] // 12,
                                min_periods=20).mean().shift(1).fillna(0.0)
        buffered = _buffer(lev.values, (buffer_frac * avg_mag).values)
        lev = pd.Series(buffered, index=df.index)

    pos = lev.shift(1).fillna(0.0)                       # trade next bar
    gross = pos * ret
    cost_per_turn = (HALF_SPREAD + extra_slip_usd) / close   # fraction per |dpos|
    dpos = pos.diff().abs().fillna(pos.abs())
    cost = dpos * cost_per_turn
    net = gross - cost

    eq_net = (1.0 + net).cumprod()
    eq_gross = (1.0 + gross).cumprod()
    return dict(net=net, gross=gross, dpos=dpos, eq_net=eq_net,
                eq_gross=eq_gross, pos=pos)


def daily_sharpe(bar_ret: pd.Series) -> float:
    eq = (1.0 + bar_ret.loc[EVAL_START:]).cumprod()
    d = eq.resample("D").last().pct_change(fill_method=None).dropna()
    return float(d.mean() / d.std() * np.sqrt(252)) if d.std() > 0 else 0.0


def maxdd(bar_ret: pd.Series) -> float:
    eq = (1.0 + bar_ret.loc[EVAL_START:]).cumprod()
    return float((eq / eq.cummax() - 1.0).min() * 100)


def summarize(name: str, df: pd.DataFrame, res: dict, tf: str,
              champ_daily: pd.Series | None) -> dict:
    net = res["net"]
    eval_net = net.loc[EVAL_START:]
    eval_gross = res["gross"].loc[EVAL_START:]
    n_days = (eval_net.index[-1] - eval_net.index[0]).days
    ann_turn = res["dpos"].loc[EVAL_START:].sum() / (n_days / 365.25)
    # daily trades proxy: position sign changes per trading day
    sign = np.sign(res["pos"].loc[EVAL_START:])
    flips = (sign != sign.shift(1)).sum()
    trades_day = flips / (n_days / (365.25 / 252))

    out = dict(name=name, tf=tf,
               sharpe_net=round(daily_sharpe(net), 3),
               sharpe_gross=round(daily_sharpe(res["gross"]), 3),
               dd_net_pct=round(maxdd(net), 1),
               ann_turnover=round(float(ann_turn), 1),
               trades_per_day=round(float(trades_day), 2),
               cagr_net_pct=round(
                   (float((1 + eval_net).cumprod().iloc[-1])
                    ** (365.25 / n_days) - 1) * 100, 1))
    if champ_daily is not None:
        eq = (1 + eval_net).cumprod()
        d = eq.resample("D").last().pct_change(fill_method=None).dropna()
        j = pd.concat([d, champ_daily], axis=1, join="inner").dropna()
        j.columns = ["cand", "champ"]
        if len(j) > 30:
            out["corr_champ"] = round(float(j["cand"].corr(j["champ"])), 2)
            # combined book: vol-match candidate to champ, then 50/50.
            cs, chs = j["cand"].std(), j["champ"].std()
            cand_m = j["cand"] * (chs / cs) if cs > 0 else j["cand"]
            for w, tag in ((0.5, "50"), (0.33, "33")):
                comb = w * cand_m + (1 - w) * j["champ"]
                out[f"combo{tag}_sharpe"] = round(
                    float(comb.mean() / comb.std() * np.sqrt(252)), 3)
            out["champ_alone_sharpe"] = round(
                float(j["champ"].mean() / j["champ"].std() * np.sqrt(252)), 3)
        else:
            out["corr_champ"] = None
    return out


# --------------------------------------------------------------------------
# champion daily returns (for correlation) — vectorized H4 long-only proxy
# --------------------------------------------------------------------------
def champion_daily_returns() -> pd.Series:
    from src.v5.xau_dual_signals import champion_signal
    h1 = load("H1")
    h4 = h1.resample("4h", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    fc = champion_signal(h4["close"])
    ret = h4["close"].pct_change().fillna(0.0)
    sig = ret.ewm(span=30, min_periods=20).std().shift(1)
    tgt = TARGET_ANN_VOL / np.sqrt(252 * 6)
    lev = (fc * tgt / sig.replace(0.0, np.nan)).clip(0, MAX_LEV).fillna(0.0)
    net = lev.shift(1).fillna(0.0) * ret
    eq = (1 + net.loc[EVAL_START:]).cumprod()
    return eq.resample("D").last().pct_change(fill_method=None).dropna()


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------
def build_signals(df: pd.DataFrame, tf: str) -> dict:
    """Return {name: (forecast, long_only)} for a timeframe."""
    c = df["close"]
    # bars-per-day scale so "fast" means ~hours..1-3 days, not weeks
    bpd = {"M15": 96, "M30": 48, "H1": 24}[tf]
    sig = {}
    # EWMAC intraday speeds (fast .. medium), in BARS
    ewmac_sets = {
        "vfast": [(8, 32), (16, 64)],
        "fast": [(16, 64), (32, 128)],
        "med": [(32, 128), (64, 256)],
    }
    ewmac_by_k = {}
    for k, pairs in ewmac_sets.items():
        fc = ewmac_fc(c, pairs)
        ewmac_by_k[k] = fc
        sig[f"ewmac_{k}_ls"] = (fc, False)
        sig[f"ewmac_{k}_lo"] = (fc, True)
    # breakout Donchian windows in bars: ~ half-day, 1 day, 2 days
    bko_by_label = {}
    for label, mult in (("halfday", 0.5), ("day", 1.0), ("2day", 2.0)):
        n = max(6, int(bpd * mult))
        fc = breakout_fc(c, [n, n * 2])
        bko_by_label[label] = fc
        sig[f"breakout_{label}_ls"] = (fc, False)
        sig[f"breakout_{label}_lo"] = (fc, True)
    # FAST-CHAMPION: champion recipe (max of trend & breakout, long-only,
    # concentrated ^1.5) but at intraday speeds -> the "small trend runner".
    def _norm(s):
        return s * (1.0 / s.abs().expanding(min_periods=120).mean().shift(1))
    for spd, ekey, bkey in (("fast", "fast", "day"), ("vfast", "vfast", "halfday")):
        ew, bk = ewmac_by_k[ekey], bko_by_label[bkey]
        mx = np.maximum(ew.clip(lower=0.0), bk.clip(lower=0.0))
        champ = (0.5 * (_norm(mx.clip(lower=0) ** 1.5) * 0.8 + 0.15)
                 + 0.5 * (_norm(bk.clip(lower=0) ** 1.5) * 0.8 + 0.15)).clip(0, 2)
        sig[f"fastchamp_{spd}_lo"] = (champ, True)
    # session ORB — data time index is UTC(+broker). London~7-8, NY~13.
    for oh, ch, tag in ((7, 20, "london"), (13, 21, "ny")):
        rb = {"M15": 4, "M30": 2, "H1": 1}[tf]   # ~1h opening range
        fc = orb_fc(df, oh, rb, ch)
        sig[f"orb_{tag}_ls"] = (fc, False)
        sig[f"orb_{tag}_lo"] = (fc, True)
    # intraday momentum (morning sign -> afternoon)
    fc = intraday_mom_fc(df, split_hour=13)
    sig["intraday_mom_ls"] = (fc, False)
    sig["intraday_mom_lo"] = (fc, True)
    return sig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="all", choices=["all", "M15", "M30", "H1"])
    ap.add_argument("--slip", type=float, default=0.0,
                    help="extra one-way slippage USD on top of half-spread")
    ap.add_argument("--buffer", type=float, default=0.0,
                    help="no-trade buffer as frac of avg target leverage")
    args = ap.parse_args()

    print(f"champion daily returns (corr reference) ...", flush=True)
    champ = champion_daily_returns()
    print(f"  champion eval Sharpe {float(champ.mean()/champ.std()*np.sqrt(252)):.3f}"
          f"  ({len(champ)} days)\n", flush=True)

    tfs = ["M15", "M30", "H1"] if args.tf == "all" else [args.tf]
    rows = []
    for tf in tfs:
        print(f"=== {tf} ===", flush=True)
        df = load(tf).loc["2015":]
        signals = build_signals(df, tf)
        for name, (fc, lo) in signals.items():
            res = backtest(df, fc, tf, long_only=lo, extra_slip_usd=args.slip,
                           buffer_frac=args.buffer)
            row = summarize(name, df, res, tf, champ)
            rows.append(row)
            print(f"  {name:22s} netSR {row['sharpe_net']:+.2f} "
                  f"turn {row['ann_turnover']:6.1f} "
                  f"corrC {str(row.get('corr_champ')):>5} "
                  f"combo50 {str(row.get('combo50_sharpe')):>6} "
                  f"(champ {row.get('champ_alone_sharpe')})", flush=True)

    out = pd.DataFrame(rows).sort_values("combo50_sharpe", ascending=False)
    dst = Path("data/v5_runs/fast-trend")
    dst.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst / "sweep_results.csv", index=False)
    cols = ["name", "tf", "sharpe_net", "ann_turnover", "corr_champ",
            "combo50_sharpe", "combo33_sharpe", "champ_alone_sharpe"]
    print(f"\nTOP 15 by COMBINED book Sharpe (champ+sleeve 50/50 vol-matched, "
          f"eval 2017+, net $0.34 + {args.slip} slip, buffer {args.buffer}):")
    print(out[cols].head(15).to_string(index=False))
    print(f"\nsaved -> {dst/'sweep_results.csv'}")


if __name__ == "__main__":
    main()
