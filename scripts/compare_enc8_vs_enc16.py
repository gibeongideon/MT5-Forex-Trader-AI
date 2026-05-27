"""
enc8 vs enc16 comparison on the same 49k-row dataset (May 2024 → May 2026).

Apples-to-apples with the Phase 9 +3.13 result.
Both configs use PredictorPipeline on the raw CSV filtered to the same date
range as the pre-built parquets — 2024-05-14 onwards.

Configs
-------
  A  enc 8-dim  / 30 epochs  (should reproduce ~+3.13)
  B  enc 16-dim / 60 epochs  (new — does bigger encoder beat +3.13?)

Usage
-----
  conda run -n envmt5 python scripts/compare_enc8_vs_enc16.py
  conda run -n envmt5 python scripts/compare_enc8_vs_enc16.py --skip-a
"""

from __future__ import annotations

import argparse
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

# ── Date filter — match exactly the 49k parquet date range ────────────────────
FILTER_START = pd.Timestamp("2024-05-14 11:15:00")

PREV_BEST     = 3.13   # Phase 9 enc8 on this same dataset
CACHE_ROOT    = Path("data/models/wf_cache_enc_compare")


def _load_raw(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df.columns = [c.lower() for c in df.columns]
    time_col = next((c for c in df.columns if "time" in c), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    df = df.sort_index()
    return df.loc[df.index >= FILTER_START]


def _load_prices(data_path: str) -> pd.DataFrame:
    prices = pd.read_csv(data_path, index_col=0)
    prices.index = pd.to_datetime(prices.index)
    prices.columns = [c.lower() for c in prices.columns]
    return prices.sort_index()


def _cfg_from_yaml(full_cfg: dict, latent_dim: int, epochs: int) -> PipelineConfig:
    cfg = PipelineConfig.from_dict(
        full_cfg.get("pipeline", {}),
        rm_cfg=full_cfg.get("risk_manager", {}),
    )
    cfg.encoder_latent_dim = latent_dim
    cfg.encoder_epochs     = epochs
    cfg.model_type         = "xgboost"   # always XGBoost — matches Phase 9 baseline
    return cfg


def run_config(label: str, cfg: PipelineConfig,
               df_raw: pd.DataFrame, prices: pd.DataFrame,
               cache_dir: Path) -> dict:
    print(f"\n{'='*62}")
    print(f"  Config {label}  —  enc{cfg.encoder_latent_dim}-dim / {cfg.encoder_epochs} epochs")
    print(f"  Dataset: {len(df_raw):,} bars  ({df_raw.index[0].date()} → {df_raw.index[-1].date()})")
    print(f"{'='*62}")
    t0 = time.time()

    cfg.wf_cache_dir = str(cache_dir)
    pipe = PredictorPipeline(cfg)
    X, y = pipe.build_features(df_raw)

    result  = pipe.walk_forward(X, y, prices)
    eq      = result.equity
    ret     = (eq.iloc[-1] / eq.iloc[0] - 1) if len(eq) > 1 else 0.0
    elapsed = time.time() - t0

    print(f"\n  Sharpe : {result.sharpe:+.2f}")
    print(f"  MaxDD  : {result.drawdown:.1f}%")
    print(f"  Return : {ret:+.1%}")
    print(f"  Trades : {len(result.trades)}")
    print(f"  Time   : {elapsed/60:.1f} min")

    return {
        "label":   label,
        "dim":     cfg.encoder_latent_dim,
        "epochs":  cfg.encoder_epochs,
        "sharpe":  result.sharpe,
        "maxdd":   result.drawdown,
        "ret":     ret,
        "trades":  len(result.trades),
        "elapsed": elapsed,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-a", action="store_true",
                   help="Skip Config A (enc8) — use if you already know it gives +3.13")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = p.parse_args()

    with open(args.config) as f:
        full_cfg = yaml.safe_load(f)

    data_path = full_cfg.get("pipeline", {}).get("data_path", "data/EURUSD_M15.csv")
    df_raw    = _load_raw(data_path)
    prices    = _load_prices(data_path)

    print(f"\nDataset after date filter: {len(df_raw):,} bars "
          f"({df_raw.index[0].date()} → {df_raw.index[-1].date()})")

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    results = []

    # ── Config A: enc 8-dim / 30 epochs ──────────────────────────────────────
    if not args.skip_a:
        cfg_a = _cfg_from_yaml(full_cfg, latent_dim=8, epochs=30)
        results.append(run_config("A", cfg_a, df_raw, prices,
                                  CACHE_ROOT / "enc8"))
    else:
        print("\n  Config A skipped (--skip-a)  [known result: Sharpe +3.13]")
        results.append({
            "label": "A", "dim": 8, "epochs": 30,
            "sharpe": 3.13, "maxdd": 13.3, "ret": 3.583,
            "trades": 524, "elapsed": 0, "_cached": True,
        })

    # ── Config B: enc 16-dim / 60 epochs ─────────────────────────────────────
    cfg_b = _cfg_from_yaml(full_cfg, latent_dim=16, epochs=60)
    results.append(run_config("B", cfg_b, df_raw, prices,
                              CACHE_ROOT / "enc16"))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n" + "="*70)
    print("  ENC8 vs ENC16 — SAME DATASET (May 2024 → May 2026, ~49k bars)")
    print("="*70)
    header = f"  {'Cfg':<4} {'Encoder':<22} {'Sharpe':>7} {'MaxDD':>7} {'Return':>8} {'Trades':>7}"
    print(header)
    print("  " + "-"*64)

    best = max(r["sharpe"] for r in results)
    for r in results:
        cached = "  (Phase 9 result)" if r.get("_cached") else ""
        marker = "  ◄ BEST" if r["sharpe"] == best and not r.get("_cached") else ""
        beats  = "  ✓ BEATS +3.13" if r["sharpe"] > PREV_BEST and not r.get("_cached") else ""
        enc_label = f"enc{r['dim']}-dim / {r['epochs']}ep"
        print(
            f"  {r['label']:<4} {enc_label:<22} "
            f"{r['sharpe']:>+7.2f} "
            f"{r['maxdd']:>6.1f}% "
            f"{r['ret']:>+7.1%} "
            f"{r['trades']:>7}"
            f"{marker}{beats}{cached}"
        )

    print("  " + "-"*64)
    print(f"  Phase 9 record (enc8, same dataset): Sharpe +{PREV_BEST:.2f}")
    print("="*70)

    enc16 = next((r for r in results if r["dim"] == 16), None)
    if enc16:
        if enc16["sharpe"] > PREV_BEST:
            print(f"\n  NEW RECORD: enc16 Sharpe {enc16['sharpe']:+.2f}  (was +{PREV_BEST:.2f})")
        else:
            delta = enc16["sharpe"] - PREV_BEST
            print(f"\n  enc16 did not beat enc8: {enc16['sharpe']:+.2f} vs +{PREV_BEST:.2f} ({delta:+.2f})")
    print()


if __name__ == "__main__":
    main()
