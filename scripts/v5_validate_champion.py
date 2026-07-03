#!/usr/bin/env python
"""Run V5 strict fold-local champion validation and write artifacts."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
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
from src.v5.candle_validation import (
    V5CandleTrailValidationConfig,
    run_candle_trail_validation,
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


def _broker_rules(symbol: str, args):
    rules = default_broker_rules_for_symbol(symbol)
    overrides = {}
    for field in [
        "spread_pips",
        "commission_pips",
        "slippage_pips",
        "entry_delay_bars",
        "max_lot",
    ]:
        value = getattr(args, field, None)
        if value is not None:
            overrides[field] = value
    if not overrides:
        return rules
    return replace(rules, **overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. EURUSD or USDJPY")
    parser.add_argument(
        "--mode",
        default="strict",
        choices=["strict", "hybrid-v2", "candle-trail"],
        help="Validation mode: strict baseline, hybrid v2 with candle features, or candle_trail replay",
    )
    parser.add_argument("--data", default=None, help="CSV path; defaults to data/<SYMBOL>_M15.csv")
    parser.add_argument("--candle-features", default=None, help="OOS candle feature parquet/CSV for hybrid-v2")
    parser.add_argument("--signals", default=None, help="OOS candle signal parquet/CSV for candle-trail")
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
    parser.add_argument("--trail-activation-pips", type=float, default=15.0)
    parser.add_argument("--trail-pips-behind", type=float, default=10.0)
    parser.add_argument("--max-bars-low", type=int, default=1)
    parser.add_argument("--max-bars-med", type=int, default=2)
    parser.add_argument("--max-bars-high", type=int, default=4)
    parser.add_argument("--spread-pips", type=float, default=None)
    parser.add_argument("--commission-pips", type=float, default=None)
    parser.add_argument("--slippage-pips", type=float, default=None)
    parser.add_argument("--entry-delay-bars", type=int, default=None)
    parser.add_argument("--max-lot", type=float, default=None)
    parser.add_argument("--max-folds", type=int, default=None, help="Limit folds for quick smoke runs")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    data_path = Path(args.data) if args.data else ROOT / "data" / f"{symbol}_M15.csv"
    rules = _broker_rules(symbol, args)

    if args.mode == "candle-trail":
        signals_path = (
            Path(args.signals)
            if args.signals
            else ROOT / "data" / "features" / f"candle_signal_{symbol}.parquet"
        )
        run_id = args.run_id or f"{symbol.lower()}-candle-trail"
        cfg = V5CandleTrailValidationConfig(
            symbol=symbol,
            data_path=data_path,
            signals_path=signals_path,
            run_id=run_id,
            artifact_root=args.artifact_root,
            requested_lot=args.lot,
            sl_pips=args.sl_pips,
            tp_pips=args.tp_pips,
            threshold=args.threshold,
            broker_rules=rules,
            trail_activation_pips=args.trail_activation_pips,
            trail_pips_behind=args.trail_pips_behind,
            max_bars_low=args.max_bars_low,
            max_bars_med=args.max_bars_med,
            max_bars_high=args.max_bars_high,
        )
        result = run_candle_trail_validation(cfg)
        print(f"V5 candle-trail artifacts: {result.run_dir}")
        print(
            f"mode={result.stats['mode']} trades={result.stats['trades']} "
            f"final_equity={result.stats['final_equity']:.2f}"
        )
        return

    run_id = args.run_id or f"{symbol.lower()}-{args.mode}-{args.model}"
    pipeline = _pipeline_config(Path(args.config), symbol, data_path, args)
    candle_features_path = None
    if args.mode == "hybrid-v2":
        candle_features_path = (
            Path(args.candle_features)
            if args.candle_features
            else ROOT / "data" / "features" / f"candle_signal_{symbol}.parquet"
        )
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
        candle_features_path=candle_features_path,
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
