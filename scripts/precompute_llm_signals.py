"""
Pre-compute LLM signals for all historical bars.

Solves the backtesting cost problem: instead of calling the Claude API
~38k times during walk-forward validation, this script calls it once per
stride bars and saves results to a parquet cache.

The LLMSignalModel then reads from this cache (no API calls during walk-forward).

Usage:
    conda activate envmt5
    # Dry run — estimate cost without calling API
    python scripts/precompute_llm_signals.py --dry-run

    # Full run (calls API)
    python scripts/precompute_llm_signals.py

    # Custom params
    python scripts/precompute_llm_signals.py \\
        --features data/features/EURUSD_M15_features.parquet \\
        --prices   data/EURUSD_M15.csv \\
        --output   data/models/llm_cache.parquet \\
        --model    claude-haiku-4-5-20251001 \\
        --n-bars   32 \\
        --stride   4

Cost estimate at stride=4:
    ~12,500 API calls × $0.00025/call (Haiku + cached system prompt) ≈ $3 total
    Runtime: ~30–60 minutes depending on rate limits

The output cache can be loaded by:
    model = LLMSignalModel(cache_path="data/models/llm_cache.parquet")
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.features.bar_tokenizer import BarTokenizer
from src.models.llm_signal_model import LLMSignalModel


def estimate_cost(n_bars: int, stride: int, model_id: str) -> None:
    n_calls = n_bars // stride
    # Approximate cost: Haiku with cached system ~$0.25/1k input tokens
    # User message ≈ 50 tokens, system cached ≈ 200 tokens (first call) then 0
    # Output ≈ 20 tokens
    input_tokens  = n_calls * 50
    output_tokens = n_calls * 20
    cost_input    = input_tokens  / 1e6 * 0.80   # Haiku $0.80/M input
    cost_output   = output_tokens / 1e6 * 4.00   # Haiku $4.00/M output
    total_cost    = cost_input + cost_output

    print(f"\n{'─' * 50}")
    print(f"  DRY RUN COST ESTIMATE")
    print(f"{'─' * 50}")
    print(f"  Bars to process   : {n_bars:,}")
    print(f"  Stride            : {stride} (call every {stride} bars)")
    print(f"  API calls needed  : {n_calls:,}")
    print(f"  Model             : {model_id}")
    print(f"  Est. input tokens : {input_tokens:,}")
    print(f"  Est. output tokens: {output_tokens:,}")
    print(f"  Est. total cost   : ${total_cost:.2f}")
    print(f"  Est. runtime      : {n_calls * 0.3 / 60:.0f}–{n_calls * 0.8 / 60:.0f} minutes")
    print(f"{'─' * 50}\n")


def main():
    p = argparse.ArgumentParser(description="Pre-compute LLM signals for all bars")
    p.add_argument("--features", default="data/features/EURUSD_M15_features.parquet")
    p.add_argument("--prices",   default="data/EURUSD_M15.csv",
                   help="Raw OHLCV CSV (needed for tokenizer context)")
    p.add_argument("--output",   default="data/models/llm_cache.parquet")
    p.add_argument("--model",    default="claude-haiku-4-5-20251001")
    p.add_argument("--n-bars",   type=int,   default=32,
                   help="Context window (bars of history per prompt)")
    p.add_argument("--stride",   type=int,   default=4,
                   help="Call LLM every N bars; interpolate between calls")
    p.add_argument("--dry-run",  action="store_true",
                   help="Print cost estimate without calling API")
    p.add_argument("--start",    default=None,
                   help="Start datetime (skip bars before this, useful for resume)")
    p.add_argument("--batch-save", type=int, default=500,
                   help="Save cache to disk every N API calls")
    args = p.parse_args()

    print("Loading data...")
    X = pd.read_parquet(args.features)
    print(f"Feature matrix: {X.shape}")

    prices_path = Path(args.prices)
    if prices_path.exists():
        prices = pd.read_csv(prices_path, index_col="time")
        prices.index = pd.to_datetime(prices.index)
        # Merge OHLCV with feature columns for richer tokenizer context
        combined = prices[["open", "high", "low", "close"]].join(
            X[["atr_14", "rsi_14", "sma_50"]].rename(columns={
                "atr_14": "atr_14", "rsi_14": "rsi_14", "sma_50": "sma_50"
            }), how="inner"
        )
    else:
        print(f"Warning: {args.prices} not found — using feature matrix only")
        combined = X.copy()

    n_bars = len(combined)
    estimate_cost(n_bars, args.stride, args.model)

    if args.dry_run:
        print("Dry run complete. Remove --dry-run to execute.")
        return

    # Load or create cache
    output_path = Path(args.output)
    if output_path.exists():
        cache = pd.read_parquet(output_path)
        print(f"Loaded existing cache: {len(cache)} entries")
    else:
        cache = pd.DataFrame(columns=["P_buy", "P_hold", "P_sell"])

    # Filter start if specified
    if args.start:
        start_dt = pd.to_datetime(args.start)
        combined = combined[combined.index >= start_dt]
        print(f"Resuming from {args.start} ({len(combined)} bars remaining)")

    # Skip bars already in cache
    already_cached = set(cache.index)
    indices = [ts for ts in combined.index if ts not in already_cached]
    # Process only at stride intervals
    stride_indices = [ts for i, ts in enumerate(indices) if i % args.stride == 0]
    print(f"Bars to process: {len(stride_indices):,} (stride={args.stride})")

    model = LLMSignalModel(
        model_id       = args.model,
        n_context_bars = args.n_bars,
        cache_path     = args.output,
    )

    tok = BarTokenizer()
    tok.fit(combined)

    new_rows = []
    call_count = 0
    errors = 0

    for i, ts in enumerate(stride_indices):
        # Build context window ending at ts
        pos = combined.index.get_loc(ts)
        if isinstance(pos, slice):
            pos = pos.start
        start_pos = max(0, pos - args.n_bars + 1)
        ctx = combined.iloc[start_pos: pos + 1]

        try:
            proba = model._call_api(ctx)
        except Exception as e:
            print(f"  Error at {ts}: {e}")
            errors += 1
            proba = np.array([1/3, 1/3, 1/3])

        new_rows.append({
            "ts":     ts,
            "P_buy":  float(proba[0]),
            "P_hold": float(proba[1]),
            "P_sell": float(proba[2]),
        })
        call_count += 1

        if (i + 1) % 50 == 0:
            win_direction = "buy" if proba[0] > proba[2] else "sell"
            print(f"  [{i+1:>5}/{len(stride_indices)}] {ts}  "
                  f"P_buy={proba[0]:.2f}  P_hold={proba[1]:.2f}  P_sell={proba[2]:.2f}"
                  f"  → {win_direction}")

        # Periodic save to disk
        if call_count % args.batch_save == 0:
            _flush(cache, new_rows, output_path)
            new_rows = []
            print(f"  Checkpoint saved ({call_count} calls so far, {errors} errors)")

    # Final save
    _flush(cache, new_rows, output_path)

    # Now interpolate / forward-fill for bars between stride points
    print("\nForward-filling between stride points...")
    cache = pd.read_parquet(output_path)
    cache = cache.sort_index()
    all_ts = combined.index
    cache = cache.reindex(all_ts).ffill().bfill()  # fill gaps
    cache.to_parquet(output_path)

    print(f"\nDone! Cache saved to {output_path}")
    print(f"  Entries : {len(cache):,}")
    print(f"  API calls: {call_count}  Errors: {errors}")


def _flush(existing: pd.DataFrame, new_rows: list, path: Path) -> None:
    if not new_rows:
        return
    new_df = pd.DataFrame(new_rows).set_index("ts")
    new_df.index.name = None
    combined = pd.concat([existing, new_df]).drop_duplicates()
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.sort_index().to_parquet(path)


if __name__ == "__main__":
    main()
