#!/usr/bin/env python
"""Run V5 strict fold-local champion validation and write artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

from src.pipeline import PipelineConfig
from src.v5.champion_validation import (
    V5ChampionValidationConfig,
    default_broker_rules_for_symbol,
    run_champion_validation,
)


def _pipeline_config(config_path: Path, symbol: str, data_path: Path, args) -> PipelineConfig:
    with config_path.open() as f:
        full = yaml.safe_load(f)
    cfg = PipelineConfig.from_dict(
        full.get("pipeline", {}),
        rm_cfg=full.get("risk_manager", {}),
    )
    cfg.data_path = str(data_path)
    cfg.model_type = args.model
    cfg.encoder_enabled = not args.no_enc
    cfg.wf_train_days = args.train_days
    cfg.wf_test_days = args.test_days
    cfg.bt_threshold = args.threshold
    cfg.bt_sl_pips = args.sl_pips
    cfg.bt_tp_pips = args.tp_pips
    cfg.bt_pip_size = 0.01 if symbol.upper().endswith("JPY") or symbol.upper() == "XAUUSD" else 0.0001
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. EURUSD or USDJPY")
    parser.add_argument("--data", default=None, help="CSV path; defaults to data/<SYMBOL>_M15.csv")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default=str(ROOT / "data" / "v5_runs"))
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--no-enc", action="store_true", help="Disable latent encoder for a faster audit run")
    parser.add_argument("--train-days", type=int, default=180)
    parser.add_argument("--test-days", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--lot", type=float, default=0.01)
    parser.add_argument("--sl-pips", type=float, default=30.0)
    parser.add_argument("--tp-pips", type=float, default=60.0)
    parser.add_argument("--max-folds", type=int, default=None, help="Limit folds for quick smoke runs")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    data_path = Path(args.data) if args.data else ROOT / "data" / f"{symbol}_M15.csv"
    run_id = args.run_id or f"{symbol.lower()}-strict-{args.model}"
    pipeline = _pipeline_config(Path(args.config), symbol, data_path, args)
    rules = default_broker_rules_for_symbol(symbol)
    cfg = V5ChampionValidationConfig(
        symbol=symbol,
        data_path=data_path,
        run_id=run_id,
        artifact_root=args.artifact_root,
        requested_lot=args.lot,
        sl_pips=args.sl_pips,
        tp_pips=args.tp_pips,
        pipeline=pipeline,
        broker_rules=rules,
        max_folds=args.max_folds,
    )
    result = run_champion_validation(cfg)
    print(f"V5 strict validation artifacts: {result.run_dir}")
    print(
        f"folds={result.stats['folds']} signals={result.stats['signals']} "
        f"trades={result.stats['trades']} final_equity={result.stats['final_equity']:.2f}"
    )


if __name__ == "__main__":
    main()
