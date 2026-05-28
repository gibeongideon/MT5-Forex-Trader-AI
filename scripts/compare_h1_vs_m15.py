"""
XGBoost + enc8 on H1 vs M15 — same date range, same SL/TP.

Runs the winning config (XGBoost + supervised enc8) on EURUSD H1 data
covering the same period as the M15 champion (May 2024 → May 2026).
SL/TP kept at 30p/60p for a direct apples-to-apples comparison.

After this we can test H1-typical wider stops (e.g. 50p/100p).

Usage
-----
  conda run -n envmt5 python scripts/compare_h1_vs_m15.py

Results vs baseline
-------------------
  M15 champion: Sharpe +3.13, MaxDD 13.3%, Return +358.3%, 524 trades
  H1 result:    printed below
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.pipeline import PredictorPipeline, PipelineConfig

# ── Same date boundary as M15 49k champion ────────────────────────────────────
FILTER_START = pd.Timestamp("2024-05-14")

# ── M15 reference ─────────────────────────────────────────────────────────────
M15_CHAMPION = dict(sharpe=3.13, maxdd=13.3, ret=3.583, trades=524,
                    label="XGBoost + enc8  M15  49k  30p/60p  [CHAMPION]")


def _load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    time_col = next(c for c in df.columns if "time" in c)
    df[time_col] = pd.to_datetime(df[time_col])
    return df.set_index(time_col).sort_index()


def _load_prices(path: str) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0)
    prices.index = pd.to_datetime(prices.index)
    prices.columns = [c.lower() for c in prices.columns]
    return prices.sort_index()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide-stops", action="store_true",
                        help="Use H1-typical stops: SL=50p TP=100p instead of 30p/60p")
    args = parser.parse_args()

    sl = 50.0 if args.wide_stops else 30.0
    tp = 100.0 if args.wide_stops else 60.0
    stop_tag = "50p/100p" if args.wide_stops else "30p/60p"
    cache_tag = "wide" if args.wide_stops else "30p"

    with open(ROOT / "config.yaml") as f:
        full_cfg = yaml.safe_load(f)

    # ── Build H1-specific PipelineConfig ──────────────────────────────────────
    cfg = PipelineConfig.from_dict(
        full_cfg.get("pipeline", {}),
        rm_cfg=full_cfg.get("risk_manager", {}),
    )
    cfg.model_type         = "xgboost"
    cfg.encoder_mode       = "supervised"
    cfg.encoder_latent_dim = 8
    cfg.encoder_epochs     = 30
    cfg.wf_cache_dir       = str(ROOT / f"data/models/wf_cache_h1_enc8_{cache_tag}")
    cfg.bt_sl_pips         = sl
    cfg.bt_tp_pips         = tp

    # ── Load H1 data, filter to same period as M15 champion ───────────────────
    h1_path = str(ROOT / "data/EURUSD_H1.csv")
    df_raw  = _load_raw(h1_path)
    df_h1   = df_raw[df_raw.index >= FILTER_START].copy()

    print(f"\nH1 dataset : {len(df_h1):,} bars "
          f"({df_h1.index[0].date()} → {df_h1.index[-1].date()})")
    print(f"M15 dataset: ~49,892 bars (2024-05-14 → 2026-05-18)  [reference]")
    print(f"\nConfig: XGBoost + supervised enc8  SL={sl:.0f}p  TP={tp:.0f}p  "
          f"({'H1-typical' if args.wide_stops else 'same as M15 champion'})")

    prices = _load_prices(str(ROOT / full_cfg["pipeline"]["data_path"]))

    # ── Run pipeline ──────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  XGBoost + enc8  H1  {df_h1.index[0].date()} → {df_h1.index[-1].date()}  SL={sl:.0f}p TP={tp:.0f}p")
    print(f"{'='*64}")
    t0 = time.time()

    pipe    = PredictorPipeline(cfg)
    X, y    = pipe.build_features(df_h1)
    result  = pipe.walk_forward(X, y, prices)

    eq      = result.equity
    ret     = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    elapsed = time.time() - t0

    print(f"\n  Sharpe : {result.sharpe:+.2f}")
    print(f"  MaxDD  : {result.drawdown:.1f}%")
    print(f"  Return : {ret:+.1%}")
    print(f"  Trades : {len(result.trades)}")
    print(f"  Time   : {elapsed/60:.1f} min")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "="*76)
    print(f"  H1 vs M15 — XGBoost + enc8, same date range (May 2024+), SL={sl:.0f}p TP={tp:.0f}p")
    print("="*76)
    print(f"  {'Timeframe':<46} {'Sharpe':>7} {'MaxDD':>7} {'Return':>8} {'Trades':>7}")
    print("  " + "-"*70)

    c = M15_CHAMPION
    print(f"  {c['label']:<46} {c['sharpe']:>+7.2f} {c['maxdd']:>6.1f}% "
          f"{c['ret']:>+7.1%} {c['trades']:>7}")
    print("  " + "-"*70)

    h1_label = f"XGBoost + enc8  H1   {len(df_h1):,} bars  {stop_tag}"
    delta    = result.sharpe - c["sharpe"]
    flag     = "  ✓ BEATS M15!" if delta > 0 else f"  ({delta:+.2f} vs M15)"
    print(f"  {h1_label:<46} {result.sharpe:>+7.2f} {result.drawdown:>6.1f}% "
          f"{ret:>+7.1%} {len(result.trades):>7}{flag}")

    print("  " + "-"*70)
    print("="*76)
    print()

    if not args.wide_stops:
        print("  NEXT STEP: run with H1-typical stops (SL=50p TP=100p)")
        print("  Command:   conda run -n envmt5 python scripts/compare_h1_vs_m15.py --wide-stops")
    print()


if __name__ == "__main__":
    main()
