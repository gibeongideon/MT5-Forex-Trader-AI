# V5 MT5 + Lumibot Profitability Plan

> Coordination document for the V5 profitability track.
> Last updated: 2026-07-02
> Status: Active implementation plan

## Goal

Create a working profitable strategy from the existing MT5 research stack while
using Lumibot as the reference for stronger lifecycle discipline, backtest
artifacts, replay quality, and broker-realistic validation.

V5 treats every high Sharpe result as untrusted until it survives leakage checks,
costs, execution assumptions, dry-run/paper trading, and live replay
reconciliation.

## Current Repo Findings

- Existing MT5 stack already has serious research infrastructure:
  `PredictorPipeline`, walk-forward backtests, candle models, hybrid v2
  candle-feature injection, CTA modules, MT5/MT4 connectors, trade journal, and
  live `PipelineBot` execution modes.
- The most promising current candidates are:
  - `candle_trail` for EURUSD/USDJPY.
  - hybrid v2 XGBoost + enc8 + candle feature injection for EURUSD/USDJPY.
  - CTA portfolio only as a later track unless the champion validation fails.
- The key technical risk is validation leakage:
  `PredictorPipeline.walk_forward()` retrains the classifier per fold, but the
  current feature/scaler/encoder stack is built before the folds. V5 must prove
  fold-local fitting before any metric is considered deployable.
- Lumibot should be used adapter-first:
  keep MT5 live trading intact, borrow Lumibot-style lifecycle and artifacts,
  then decide later whether a full Lumibot runtime migration is worth it.

## Implemented V5 Foundation

- [x] Added `src/v5/validation.py`.
  - Past-only feature builder guard.
  - Forward-return label guard.
  - Fold-local fit-record guard for scaler/encoder/model audits.
  - Candle OOS prediction guard.
  - Broker execution rules for pip size, entry delay, costs, lot rounding, and
    lot caps.
- [x] Added `src/v5/artifacts.py`.
  - Writes Lumibot-style run bundles:
    `settings.json`, `stats.json`, `trades.csv`, `equity.csv`, `folds.csv`,
    and `reconciliation.json`.
- [x] Added `src/v5/folds.py`.
  - Defines strict half-open expanding fold windows.
  - Skips partial final test windows so champion reruns compare like-for-like
    folds.
  - Emits fit-window records that feed the train-only validation guard.
- [x] Added `src/v5/replay.py`.
  - Replays an explicit signal frame with broker-realistic entry delay, spread,
    commission, slippage, SL/TP, lot rounding, and max-lot rules.
  - Produces trades, equity, and settings ready for `V5ArtifactWriter`.
- [x] Added `src/v5/strict_pipeline.py`.
  - Runs fold-local feature fitting, optional encoder fitting, optional candle
    tokenizer fitting, classifier training, and test-signal generation.
  - Emits fit-window records for `feature_scaler`, optional `encoder`, optional
    `candle_tokenizer`, and `classifier`.
- [x] Added `src/v5/champion_validation.py` and `scripts/v5_validate_champion.py`.
  - Loads real symbol CSV data.
  - Runs the strict fold-local pipeline.
  - Replays signals through broker-realistic rules.
  - Writes `data/v5_runs/<run_id>/` artifacts for review.
- [x] Added focused tests:
  - `tests/test_v5_validation.py`
  - `tests/test_v5_artifacts.py`
  - `tests/test_v5_folds.py`
  - `tests/test_v5_replay.py`
  - `tests/test_v5_strict_pipeline.py`
  - `tests/test_v5_champion_validation.py`

## Phase 1: Validation Hardening

- [x] Establish reusable validation helpers.
- [x] Add strict fold-window scaffolding for the champion path:
  - Use half-open train/test fold windows.
  - Record component fit windows for scaler/encoder/classifier audits.
- [x] Add strict fold-local pipeline execution scaffold for the champion path:
  - Fit `FeaturePipeline` scaler only on each fold training window.
  - Fit `LatentEncoder` only on each fold training window.
  - Generate test features by transform-only behavior inside the fold.
  - Train classifier only on fold-local training features.
  - Record per-component fit windows and pass
    `assert_fold_fit_records_are_train_only()`.
- [x] Add a strict champion validation script that writes review artifacts.
- [ ] Run strict fold-local champion validations on real EURUSD/USDJPY data and
  compare against the documented leaderboard.
- [ ] Add exact reproduction modes for current documented champions:
  - EURUSD `candle_trail`
  - USDJPY `candle_trail`
  - EURUSD hybrid v2
  - USDJPY hybrid v2
- [x] Save strict validation runs under `data/v5_runs/<run_id>/` using
  `V5ArtifactWriter`.
- [ ] Mark the old global-feature walk-forward numbers as research-only until
  strict fold-local results exist.

### Strict Validation Evidence

These are bounded smoke/diagnostic runs, not final champion claims.

| Run ID | Symbol | Mode | Folds | Signals | Trades | Final Equity | Return | MaxDD | Win Rate | Verdict |
| ------ | ------ | ---- | ----- | ------- | ------ | ------------ | ------ | ----- | -------- | ------- |
| `eurusd-strict-smoke` | EURUSD | XGBoost, no encoder, 30d/10d, 1 fold | 1 | 675 | 1 | 10005.82 | +0.058% | 0.00% | 100.0% | smoke only |
| `eurusd-strict-xgb-5fold-noenc` | EURUSD | XGBoost, no encoder, 180d/30d, 5 folds | 5 | 10245 | 75 | 10004.50 | +0.045% | 0.45% | 36.0% | flat baseline |
| `usdjpy-strict-xgb-5fold-noenc` | USDJPY | XGBoost, no encoder, 180d/30d, 5 folds | 5 | 10240 | 270 | 9996.40 | -0.036% | 0.78% | 35.2% | weak baseline |

Interpretation: strict fold-local XGBoost without the encoder does not reproduce
the leaderboard edge. This is expected if the real edge depends on enc8 and/or
candle feature injection. Next validations must test encoder-enabled and hybrid
paths before judging the champion claims.

## Phase 2: Broker-Realistic Simulation

- [x] Add one-bar entry delay support to V5 replay.
- [x] Apply reusable broker cost and lot rules:
  - spread pips
  - commission pips
  - slippage pips
  - symbol pip size
  - lot minimum
  - lot step
  - max lot
- [ ] Match MT5/HFM behavior as closely as possible:
  - symbol suffix resolution
  - SL/TP levels
  - volume rounding
  - magic-number isolation
  - session rules
- [ ] Run sensitivity checks:
  - worse spread
  - worse slippage
  - lower confidence thresholds
  - delayed entry
  - disabled candle feature injection fallback

## Phase 3: Lumibot Bridge, Adapter First

- [ ] Add a thin strategy lifecycle wrapper around the existing MT5 strategy
  components:
  - `initialize`
  - `on_iteration`
  - `create_order_intent`
  - `record_fill`
  - `record_stats`
- [ ] Keep `PipelineBot` as the live runtime in this phase.
- [ ] Emit Lumibot-style artifacts through `V5ArtifactWriter`.
- [ ] Keep full Lumibot broker/data adapter work out of scope until the
  validated edge survives Phase 1 and Phase 2.

## Phase 4: Paper Trading Gate

- [ ] Run dry-run first with a dedicated run id and artifact directory.
- [ ] Run demo-live with small lots and a dedicated magic number.
- [ ] Compare expected signals/orders to actual broker actions:
  - symbol
  - side
  - lot
  - SL
  - TP
  - magic
  - close reason
- [ ] Reconcile broker fills and realized PnL against replay within documented
  tolerance.
- [ ] Do not risk real capital until this gate passes.

## Phase 5: Migration Decision

- [ ] If adapter-first is sufficient, keep the MT5 runtime and continue adding
  Lumibot-compatible reports.
- [ ] If Lumibot runtime provides enough operational value, build a custom MT5
  broker/data adapter and port only the validated strategy.

## Acceptance Criteria

- A champion strategy is deployable only when:
  - all V5 validation tests pass;
  - strict fold-local results remain profitable;
  - broker-realistic replay remains profitable;
  - paper/demo results reconcile with expected signals, fills, and PnL;
  - all artifacts are written to `data/v5_runs/<run_id>/`;
  - real-capital risk is explicitly approved after the evidence is reviewed.

## Commands

```bash
pytest tests/test_v5_validation.py tests/test_v5_artifacts.py tests/test_v5_folds.py tests/test_v5_replay.py tests/test_v5_strict_pipeline.py tests/test_v5_champion_validation.py -q
conda run -n envmt5 python scripts/v5_validate_champion.py --symbol EURUSD --data data/EURUSD_M15.csv --run-id eurusd-strict-smoke --no-enc --train-days 30 --test-days 10 --model xgboost --max-folds 1
conda run -n envmt5 python scripts/v5_validate_champion.py --symbol EURUSD --data data/EURUSD_M15.csv --run-id eurusd-strict-xgb-5fold-noenc --no-enc --train-days 180 --test-days 30 --model xgboost --max-folds 5
conda run -n envmt5 python scripts/v5_validate_champion.py --symbol USDJPY --data data/USDJPY_M15.csv --run-id usdjpy-strict-xgb-5fold-noenc --no-enc --train-days 180 --test-days 30 --model xgboost --max-folds 5
```

Future heavier validation commands should be added here as they are built, with
the exact run id and output artifact path.
