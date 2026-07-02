#!/usr/bin/env python
"""Build V5 fold-local OOS candle probability files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.v5.candle_oos import V5CandleOOSConfig, generate_candle_oos_predictions


def _load_ohlcv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [c.lower() for c in frame.columns]
    time_col = next((c for c in frame.columns if "time" in c or c in {"date", "datetime"}), None)
    if time_col is not None:
        frame[time_col] = pd.to_datetime(frame[time_col])
        frame = frame.set_index(time_col)
    else:
        frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True, help="Symbol, e.g. EURUSD or USDJPY")
    parser.add_argument("--data", default=None, help="CSV path; defaults to data/<SYMBOL>_M15.csv")
    parser.add_argument("--out", default=None, help="Output parquet path")
    parser.add_argument("--model", default="catboost")
    parser.add_argument("--train-days", type=int, default=120)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--no-enc", action="store_true")
    parser.add_argument("--encoder-epochs", type=int, default=30)
    args = parser.parse_args()

    symbol = args.symbol.upper()
    data_path = Path(args.data) if args.data else ROOT / "data" / f"{symbol}_M15.csv"
    out_path = (
        Path(args.out)
        if args.out
        else ROOT / "data" / "features" / f"candle_signal_v5_{symbol}.parquet"
    )
    cfg = V5CandleOOSConfig(
        symbol=symbol,
        model_type=args.model,
        train_days=args.train_days,
        test_days=args.test_days,
        max_folds=args.max_folds,
        encoder_enabled=not args.no_enc,
        encoder_epochs=args.encoder_epochs,
    )

    raw = _load_ohlcv(data_path)
    result = generate_candle_oos_predictions(raw, cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(out_path)
    print(f"V5 candle OOS predictions: {out_path}")
    print(f"folds={len(result.folds)} rows={len(result.predictions)}")
    if len(result.predictions):
        n_signals = (
            (result.predictions["candle_p_buy"] >= 0.60)
            | (result.predictions["candle_p_sell"] >= 0.60)
        ).sum()
        print(f"signal_rows={int(n_signals)}")


if __name__ == "__main__":
    main()
