# MT5 V5 Profitability Track

This branch is focused on the V5 MT5 + Lumibot-style validation path.

The goal is not to preserve every research experiment. The goal is to validate
EURUSD/USDJPY champion candidates with strict fold-local execution,
broker-realistic replay, clear artifacts, and paper/live reconciliation before
any real-capital deployment.

Read `V5_RESEARCH_AUDIT.MD` before trusting any Sharpe table. The 2026-07-04
audit fixed a remaining multi-timeframe candle feature look-ahead, so candle
signal files generated before that fix are research-only until regenerated.

## Primary Workflow

```bash
pytest tests/test_v5_validation.py tests/test_v5_artifacts.py tests/test_v5_folds.py tests/test_v5_replay.py tests/test_v5_strict_pipeline.py tests/test_v5_champion_validation.py -q
conda run -n envmt5 python scripts/v5_validate_champion.py --symbol EURUSD --data data/EURUSD_M15.csv --run-id eurusd-strict-xgb-5fold-noenc --no-enc --train-days 180 --test-days 30 --model xgboost --max-folds 5
conda run -n envmt5 python scripts/v5_build_candle_oos.py --symbol EURUSD --data data/EURUSD_M15.csv --out data/features/candle_signal_v5_EURUSD_post_mtf_fix.parquet --train-days 120 --test-days 60 --encoder-epochs 30
conda run -n envmt5 python scripts/v5_validate_champion.py --mode hybrid-v2 --symbol EURUSD --data data/EURUSD_M15.csv --candle-features data/features/candle_signal_v5_EURUSD_post_mtf_fix.parquet --run-id eurusd-hybrid-v2-strict-post-mtf-fix --train-days 180 --test-days 30 --model xgboost --max-folds 5
conda run -n envmt5 python scripts/v5_validate_champion.py --mode candle-trail --symbol EURUSD --data data/EURUSD_M15.csv --signals data/features/candle_signal_v5_EURUSD_post_mtf_fix.parquet --run-id eurusd-candle-trail-v5-post-mtf-fix --threshold 0.60 --sl-pips 10 --tp-pips 30
```

## Kept Entry Points

- `V5_PLAN.MD`: coordination plan and evidence log.
- `V5_RESEARCH_AUDIT.MD`: consolidated v4/v5 research audit and corrected
  evidence status.
- `src/v5/`: strict folds, replay, artifacts, and champion validation.
- `scripts/v5_validate_champion.py`: V5 validation command.
- `scripts/v5_build_candle_oos.py`: fold-local candle probability generation.
- `src/pipeline.py`: current predictor pipeline used by validation/live paths.
- `src/bots/pipeline_bot.py`: existing MT5 live runner, kept during adapter-first migration.
- `scripts/build_candle_features.py`, `scripts/train_candle_model.py`,
  `scripts/backtest_candle_trail.py`, `scripts/tune_candle_trail.py`,
  `scripts/verify_candle_oos.py`: candle champion support.

## Lumibot Direction

V5 is adapter-first. We borrow Lumibot-style lifecycle discipline, artifacts,
replay quality, and reconciliation checks before replacing the MT5 runtime.
