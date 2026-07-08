"""Rolling GOLD-SILVER spread backtest — adaptive beta + z-score mean reversion.

Motivation: SILVER is gold's strongest, most stable co-mover (daily-return corr
0.79, 126d rolling corr std only 0.06, positive 100% of the sample). The *static*
gold/silver spread is NOT cointegrated (Engle-Granger p~0.40), so we trade a
ROLLING-hedge spread: re-estimate beta_t on a trailing window and mean-revert the
z-score of the residual.

No-lookahead discipline:
  * beta_t uses a trailing OLS window ending at t-1 (shifted).
  * z_t uses the rolling mean/std of the spread up to t.
  * positions are applied to NEXT bar's returns (entry-delay = 1 bar).

Pre-registered run:
    python scripts/v5_gold_silver_spread_backtest.py --run-id gs-spread-v5

Stress runs (report all, never best-only):
    --cost-bps 4          # double per-leg cost
    --eval-start 2017-01-01   # recent subsample
    --beta-mode static        # naive fixed-beta comparison (expected worse)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cta.bootstrap import block_bootstrap_sharpe
from src.evaluation.metrics import max_drawdown, sharpe_ratio, sortino_ratio

PPY = 252


def load_close(data_dir: str, ticker: str) -> pd.Series:
    df = pd.read_csv(Path(data_dir) / f"{ticker}_D1_long.csv", parse_dates=["time"])
    s = df.set_index("time")["close"].sort_index()
    return s[s > 0]


def rolling_beta(g: pd.Series, s: pd.Series, window: int, mode: str) -> pd.Series:
    """Hedge ratio beta_t s.t. spread = g - beta*s, using data up to t-1.

    rolling_ols: Cov(g,s)/Var(s) over trailing `window`, shifted 1 bar.
    static:      single full-sample OLS slope (naive benchmark).
    """
    if mode == "static":
        beta = np.polyfit(s.values, g.values, 1)[0]
        return pd.Series(beta, index=g.index)
    cov = g.rolling(window).cov(s)
    var = s.rolling(window).var()
    beta = (cov / var).shift(1)          # only info available before t
    return beta


def backtest(g: pd.Series, s: pd.Series, cfg: dict) -> dict:
    df = pd.concat([np.log(g).rename("g"), np.log(s).rename("s")], axis=1).dropna()
    df["beta"] = rolling_beta(df["g"], df["s"], cfg["beta_window"], cfg["beta_mode"])
    df = df.dropna()

    df["spread"] = df["g"] - df["beta"] * df["s"]
    mu = df["spread"].rolling(cfg["z_window"]).mean()
    sd = df["spread"].rolling(cfg["z_window"]).std()
    df["z"] = (df["spread"] - mu) / sd
    df = df.dropna()

    # ---- stateful mean-reversion signal (long spread when z very negative) ----
    entry, exit_, stop = cfg["entry_z"], cfg["exit_z"], cfg["stop_z"]
    pos = np.zeros(len(df))
    z = df["z"].values
    cur = 0
    for i in range(len(df)):
        if cur == 0:
            if z[i] <= -entry:
                cur = 1                      # long spread (long gold / short silver)
            elif z[i] >= entry:
                cur = -1                     # short spread
        elif cur == 1:
            if z[i] >= -exit_ or z[i] <= -stop:
                cur = 0
        elif cur == -1:
            if z[i] <= exit_ or z[i] >= stop:
                cur = 0
        pos[i] = cur
    df["pos"] = pos

    # ---- pnl: apply position to NEXT bar's dollar-neutral spread return ----
    rg = df["g"].diff()                       # log return gold (~pct)
    rs = df["s"].diff()
    beta_lag = df["beta"].shift(1)
    spread_ret = rg - beta_lag * rs           # 1u long gold, beta short silver
    gross = df["pos"].shift(1) * spread_ret   # entry-delay 1 bar

    # ---- costs: both legs traded on any position change ----
    turnover = df["pos"].diff().abs().fillna(df["pos"].abs())
    leg_notional = 1.0 + df["beta"].abs()     # gold leg + silver leg
    cost = turnover * (cfg["cost_bps"] / 1e4) * leg_notional
    net = (gross - cost).fillna(0.0)

    out = df.copy()
    out["gross"] = gross
    out["net"] = net
    if cfg.get("eval_start"):
        out = out.loc[cfg["eval_start"]:]
    return {"df": out}


def summarize(out: pd.DataFrame, label: str) -> dict:
    net = out["net"].fillna(0.0)
    equity = (1.0 + net).cumprod()
    daily = net.copy()
    shp = float(daily.mean() / daily.std() * np.sqrt(PPY)) if daily.std() > 0 else 0.0
    ci_lo, ci_hi = block_bootstrap_sharpe(daily.values)
    trades = int((out["pos"].diff().abs() > 0).sum())
    exposure = float((out["pos"] != 0).mean())
    total_ret = float(equity.iloc[-1] - 1.0)
    years = (out.index[-1] - out.index[0]).days / 365.25
    cagr = float(equity.iloc[-1] ** (1 / years) - 1.0) if years > 0 else 0.0
    wins = net[out["pos"].shift(1) != 0]
    wr = float((wins > 0).mean()) if len(wins) else 0.0
    yearly = (1.0 + net).groupby(out.index.year).prod() - 1.0
    return {
        "label": label, "sharpe": shp, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "sortino": sortino_ratio(equity), "maxdd": max_drawdown(equity),
        "cagr": cagr, "total_ret": total_ret, "trades": trades,
        "exposure": exposure, "win_rate": wr, "years": years,
        "n": len(net), "yearly": yearly, "equity": equity,
    }


def print_summary(r: dict) -> None:
    print(f"\n=== {r['label']} ===")
    print(f"  span            {r['n']} bars  ({r['years']:.1f} yrs)")
    print(f"  Sharpe (ann)    {r['sharpe']:.2f}   95% CI [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]")
    print(f"  Sortino         {r['sortino']:.2f}")
    print(f"  CAGR            {r['cagr']*100:.2f}%   total {r['total_ret']*100:.1f}%")
    print(f"  Max drawdown    {r['maxdd']:.1f}%")
    print(f"  Trades          {r['trades']}   time-in-market {r['exposure']*100:.0f}%   win-rate {r['win_rate']*100:.0f}%")
    yr = r["yearly"]
    print("  Yearly:         " + "  ".join(f"{y}:{v*100:+.1f}%" for y, v in yr.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default="gs-spread-v5")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--gold", default="GOLD")
    ap.add_argument("--silver", default="SILVER")
    ap.add_argument("--beta-window", type=int, default=60)
    ap.add_argument("--z-window", type=int, default=20)
    ap.add_argument("--entry-z", type=float, default=2.0)
    ap.add_argument("--exit-z", type=float, default=0.5)
    ap.add_argument("--stop-z", type=float, default=4.0)
    ap.add_argument("--cost-bps", type=float, default=2.0, help="per-leg cost, bps of notional")
    ap.add_argument("--beta-mode", choices=["rolling_ols", "static"], default="rolling_ols")
    ap.add_argument("--eval-start", default=None)
    args = ap.parse_args()

    cfg = dict(beta_window=args.beta_window, z_window=args.z_window,
               entry_z=args.entry_z, exit_z=args.exit_z, stop_z=args.stop_z,
               cost_bps=args.cost_bps, beta_mode=args.beta_mode,
               eval_start=args.eval_start)

    g = load_close(args.data_dir, args.gold)
    s = load_close(args.data_dir, args.silver)
    print(f"[{args.run_id}] {args.gold}~{args.silver}  beta={args.beta_mode}(w{args.beta_window}) "
          f"z(w{args.z_window}) entry±{args.entry_z} exit±{args.exit_z} stop±{args.stop_z} "
          f"cost {args.cost_bps}bps/leg")

    res = backtest(g, s, cfg)
    print_summary(summarize(res["df"], f"PRIMARY {args.beta_mode}"))


if __name__ == "__main__":
    main()
