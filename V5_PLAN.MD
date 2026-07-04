# V5 MT5 + Lumibot Profitability Plan

> Coordination document for the V5 profitability track.
> Last updated: 2026-07-04
> Status: Active implementation plan

> 2026-07-04 audit correction: `V5_RESEARCH_AUDIT.MD` found and fixed a
> remaining multi-timeframe candle feature look-ahead in V5 fold-local candle
> OOS generation. Candle-trail Sharpe artifacts generated before this fix are
> research-only and must be regenerated before being used as proof.

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

- [x] Simplified the `V5-lumibot` branch entrypoints.
  - Removed obvious root clutter and tracked CatBoost scratch files.
  - Removed old compare/backtest/train/probe scripts that were research
    experiments rather than V5/Lumibot-style workflow.
  - Kept only the V5 validator, candle champion utilities, current live runner
    helpers, and retraining/evaluation scripts that still support the adapter
    first path.
- [x] Removed legacy compatibility shims and stale research docs.
  - Top-level `src/*.py` compatibility shims were removed; imports now use the
    package paths directly.
  - Old research notes and data-side experiment summaries were removed from the
    branch root so `README.MD`, `ARCHITECTURE.MD`, and this plan remain the
    main documentation surface.
  - The old live MT5 bridge test was removed from the default test suite because
    it requires a running external MT5/Wine service.
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
  - Supports `strict`, `hybrid-v2`, and `candle-trail` validation modes.
  - Injects OOS candle feature parquet files for hybrid v2 validation.
  - Replays OOS candle probability signals through a V5 candle-trail validator.
  - Replays signals through broker-realistic rules.
  - Supports CLI overrides for spread, commission, slippage, entry delay, and
    max lot so sensitivity runs are reproducible.
  - Writes `data/v5_runs/<run_id>/` artifacts for review.
- [x] Fixed candle-trail replay to actually honor `entry_delay_bars`.
  - Earlier candle-trail artifacts recorded entry delay in settings but opened
    positions on the signal bar.
  - The current enc30 and sensitivity tables below were regenerated after the
    fix.
- [x] Added offline MT5/HFM broker-profile checks.
  - Resolve intended broker symbol/suffix before artifact writing.
  - Annotate replay trades with broker symbol and magic number.
  - Validate symbol, magic-number isolation, volume bounds/step, and initial
    SL/TP distance.
  - Write broker-profile results into `reconciliation.json`.
- [x] Added focused tests:
  - `tests/test_v5_validation.py`
  - `tests/test_v5_artifacts.py`
  - `tests/test_v5_folds.py`
  - `tests/test_v5_replay.py`
  - `tests/test_v5_strict_pipeline.py`
  - `tests/test_v5_champion_validation.py`
  - `tests/test_v5_validate_champion_cli.py`
- [x] Added a completed-bar causality test for V5 candle extra features.
  - Mutating future M15 rows inside the same 1H/4H bin no longer changes the
    current candle feature row.

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
- [x] Run strict fold-local baseline validations on real EURUSD/USDJPY data and
  compare against the documented leaderboard.
- [x] Add exact reproduction modes for current documented champions:
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
| `eurusd-candle-trail-v5` | EURUSD | Candle-trail replay from OOS candle probabilities | n/a | n/a | 996 | 10490.15 | +4.902% | 0.056% | 77.8% | promising, research-only |
| `usdjpy-candle-trail-v5` | USDJPY | Candle-trail replay from OOS candle probabilities | n/a | n/a | 3000 | 12138.55 | +21.386% | 0.083% | 75.0% | promising, research-only |

Interpretation: strict fold-local XGBoost without the encoder does not reproduce
the leaderboard edge. This is expected if the real edge depends on enc8 and/or
candle feature injection. Next validations must test encoder-enabled and hybrid
paths before judging the champion claims.

Latest candle-trail interpretation: the primary candle-trail candidate remains
profitable under the V5 artifact/replay path, but this still consumes existing
OOS candle probability files. It is not deployable until the candle probability
generation itself is re-audited fold-by-fold, broker session behavior is matched,
and dry-run/demo reconciliation passes.

### Candle OOS Audit Evidence

The existing `scripts/verify_candle_oos.py` audit was run for both symbols. It
reloads cached CatBoost fold models and predicts each fold's OOS window only.
This supports that the CatBoost fold models are OOS, but it is not a complete
leakage clearance because the script loads the final deployed encoder, which was
trained on all data.

| Symbol | Cached Folds | Verified Sharpe | Reported v3 WF Sharpe | Trades | Win Rate | MaxDD | Net PnL | Verdict |
| ------ | ------------ | --------------- | --------------------- | ------ | -------- | ----- | ------- | ------- |
| EURUSD | 13 | +12.887 | +7.118 | 1,092 | 78.3% | 5.7% | +56,145.9% | CatBoost OOS confirmed, encoder overlap inflates |
| USDJPY | 13 | +18.602 | +14.414 | 3,241 | 71.2% | 18.5% | extremely inflated | CatBoost OOS confirmed, encoder overlap inflates |

Next audit requirement: rebuild candle OOS probability generation with
fold-local scaler and fold-local encoder fitting, then re-run candle-trail replay
from that regenerated signal file.

Implementation status: `src/v5/candle_oos.py` and
`scripts/v5_build_candle_oos.py` now provide the fold-local candle OOS generation
path. The next evidence run should generate `data/features/candle_signal_v5_*`
with encoder enabled, then replay `candle_trail` against those regenerated
signals.

Fast fold-local encoder audit (`encoder_epochs=3`) completed on 2026-07-03.
This run fits scaler, encoder, and candle classifier inside each fold before
emitting OOS candle probabilities. It is not the final 30-epoch champion setting,
but it removes the known full-data encoder overlap from the prior audit.

| Run ID | Symbol | Signal File | Folds | OOS Rows | Signal Rows | Trades | Final Equity | Return | MaxDD | Win Rate | Verdict |
| ------ | ------ | ----------- | ----- | -------- | ----------- | ------ | ------------ | ------ | ----- | -------- | ------- |
| `eurusd-candle-trail-v5-enc3` | EURUSD | `data/features/candle_signal_v5_EURUSD_enc3.parquet` | 12 | 48,941 | 176 | 149 | 10177.87 | +1.779% | 0.023% | 79.9% | survives clean fast audit |
| `usdjpy-candle-trail-v5-enc3` | USDJPY | `data/features/candle_signal_v5_USDJPY_enc3.parquet` | 12 | 48,936 | 384 | 353 | 10673.93 | +6.739% | 0.035% | 80.5% | survives clean fast audit |

Interpretation: removing the full-data encoder overlap reduces the apparent
edge sharply, but the candle-trail candidate remains profitable in the fast
fold-local audit. The next evidence step is the same run with the intended
30-epoch encoder, followed by sensitivity tests on spread/slippage/threshold.

Full fold-local encoder audit (`encoder_epochs=30`) completed on 2026-07-03.
This uses the intended candle encoder training length while keeping scaler,
encoder, and candle classifier fold-local.

Audit correction on 2026-07-04: these enc30 results were generated before the
completed-bar MTF candle feature fix. They remain useful as historical
comparison points, but they are no longer accepted as leakage-cleared proof.
Regenerate the candle OOS signal files and all derived replays before promoting
any Sharpe claim.

| Run ID | Symbol | Signal File | Folds | OOS Rows | Signal Rows | Trades | Final Equity | Return | MaxDD | Win Rate | Sharpe | Daily Sharpe | Verdict |
| ------ | ------ | ----------- | ----- | -------- | ----------- | ------ | ------------ | ------ | ----- | -------- | ------ | ------------ | ------- |
| `eurusd-candle-trail-v5-enc30` | EURUSD | `data/features/candle_signal_v5_EURUSD_enc30.parquet` | 12 | 48,941 | 254 | 202 | 10082.01 | +0.820% | 0.071% | 57.4% | 2.152 | 2.504 | survives full clean audit, thinner edge |
| `usdjpy-candle-trail-v5-enc30` | USDJPY | `data/features/candle_signal_v5_USDJPY_enc30.parquet` | 12 | 48,936 | 703 | 553 | 10352.48 | +3.525% | 0.085% | 62.9% | 4.469 | 4.305 | survives full clean audit |

Enc30 vs enc3: enc30 increases signal/trade count and slightly improves final
equity before delayed-entry correction, while lowering win rate. After the
delayed-entry correction, the edge remains positive but materially smaller,
especially for EURUSD.

Hybrid-v2 status: the runner is implemented, but local full/fold diagnostics were
interrupted because XGBoost/encoder training was too slow for the interactive
turn. Run hybrid-v2 as a longer job or add a faster validation profile before
using hybrid claims as evidence.

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
- [x] Expose broker realism overrides in `scripts/v5_validate_champion.py`:
  - `--spread-pips`
  - `--commission-pips`
  - `--slippage-pips`
  - `--entry-delay-bars`
  - `--max-lot`
- [x] Match first-pass MT5/HFM replay behavior:
  - symbol suffix resolution
  - SL/TP levels
  - volume rounding
  - magic-number isolation
- [ ] Match remaining MT5/HFM behavior:
  - session rules
- [x] Run first candle-trail sensitivity checks:
  - worse spread
  - worse slippage
  - higher confidence threshold
- [x] Run delayed-entry sensitivity with two-bar entry delay.
- [ ] Run remaining sensitivity checks:
  - lower confidence thresholds
  - disabled candle feature injection fallback
- [x] Add a live dry-run gate watcher for the USDJPY candidate.
  - `scripts/v5_watch_live_gate.py` waits for the first journaled dry-run
    order intent for a run id.
  - Once an intent exists, it runs strict `--live-current` validation through
    `scripts/v5_reconcile_dry_run.py`.
  - This prevents manual terminal watching from becoming part of the V5 gate.

### Phase 2 Sensitivity Evidence

All runs use fold-local enc30 candle probabilities and the V5 candle-trail
replay path. The `costx` runs use spread `2.0` pips and slippage `0.5` pips.
The `th065` runs use threshold `0.65` with default broker costs.
The `delay2` runs use a two-bar entry delay. Baseline uses the configured
one-bar entry delay.

| Run ID | Symbol | Stress | Trades | Final Equity | Return | MaxDD | Win Rate | Sharpe | Daily Sharpe | Verdict |
| ------ | ------ | ------ | ------ | ------------ | ------ | ----- | -------- | ------ | ------------ | ------- |
| `eurusd-candle-trail-v5-enc30` | EURUSD | baseline, delay 1 | 202 | 10082.01 | +0.820% | 0.071% | 57.4% | 2.152 | 2.504 | survives, thin edge |
| `eurusd-candle-trail-v5-enc30-costx` | EURUSD | spread/slippage worse | 203 | 10029.97 | +0.300% | 0.166% | 45.8% | 0.801 | 1.152 | barely survives |
| `eurusd-candle-trail-v5-enc30-th065` | EURUSD | threshold 0.65 | 117 | 10090.96 | +0.910% | 0.069% | 63.2% | 2.310 | 2.460 | better EURUSD filter |
| `eurusd-candle-trail-v5-enc30-delay2` | EURUSD | delay 2 | 191 | 10006.46 | +0.065% | 0.173% | 46.6% | 0.223 | 0.032 | effectively flat |
| `usdjpy-candle-trail-v5-enc30` | USDJPY | baseline, delay 1 | 553 | 10352.48 | +3.525% | 0.085% | 62.9% | 4.469 | 4.305 | baseline survives |
| `usdjpy-candle-trail-v5-enc30-costx` | USDJPY | spread/slippage worse | 553 | 10211.45 | +2.115% | 0.278% | 56.8% | 2.769 | 2.929 | survives, compressed edge |
| `usdjpy-candle-trail-v5-enc30-th065` | USDJPY | threshold 0.65 | 338 | 10296.30 | +2.963% | 0.060% | 68.0% | 4.144 | 4.303 | fewer trades, cleaner |
| `usdjpy-candle-trail-v5-enc30-delay2` | USDJPY | delay 2 | 522 | 10077.31 | +0.773% | 0.115% | 53.1% | 1.520 | 1.403 | still positive, heavily reduced |

Interpretation: after fixing delayed-entry execution, both candidates remain
positive at one-bar delay, but the edge is much smaller than the immediate-entry
artifact implied. USDJPY is still the stronger candidate. EURUSD should be
treated as fragile unless threshold filtering or broker replay improves it.

### MT5/HFM Broker-Profile Evidence

These are still research replays, not paper/live reconciliation. They prove the
replay artifacts can be mapped to the intended HFM-style broker symbol and
service magic numbers before dry-run.

| Run ID | Symbol | Broker Symbol | Magic | Threshold | Trades | Return | Sharpe | Checks | Verdict |
| ------ | ------ | ------------- | ----- | --------- | ------ | ------ | ------ | ------ | ------- |
| `usdjpy-candle-trail-v5-enc30-hfm` | USDJPY | `USDJPY.Z` | 20260103 | 0.60 | 553 | +3.525% | 4.469 | symbol, magic, volume, SL/TP distance pass | primary paper candidate |
| `eurusd-candle-trail-v5-enc30-hfm` | EURUSD | `EURUSD.Z` | 20260102 | 0.65 | 117 | +0.910% | 2.310 | symbol, magic, volume, SL/TP distance pass | secondary/fragile |

### Leakage/Sharpe Proof Evidence

`data/v5_runs/usdjpy-candle-trail-v5-enc30-hfm/leakage_proof.json` was generated
for the primary USDJPY candidate.

Checks passed:

- OOS candle probability metadata validates with
  `assert_candle_predictions_are_oos()`.
- Signal file hash:
  `b4a0adffbab3cc2e89275a83d47bebc7b3051fee6545147e4cfeb2ec16ccca72`.
- 48,936 OOS probability rows across 12 folds.
- No duplicate prediction timestamps.
- Every prediction timestamp is inside its fold's test window and outside its
  train window.
- Reported Sharpe `4.469232139163544` recomputes from `equity.csv` as
  `4.4692321391635295` with delta `1.42e-14`.
- Stress evidence remains positive:
  - cost stress Sharpe `2.769`, return `+2.115%`;
  - threshold 0.65 Sharpe `4.144`, return `+2.963%`;
  - two-bar delay Sharpe `1.520`, return `+0.773%`.

Interpretation: this supports that the USDJPY Sharpe is coming from fold-local
out-of-sample candle probabilities and the persisted replay equity curve, not
from a full-data encoder or in-sample candle prediction file. It is still not a
live-profit claim until dry-run and demo-live reconciliation pass.

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

- [x] Add dry-run journal reconciliation tooling.
  - `TradeJournal` now stores `magic`, `run_id`, and `dry_run`.
  - `PipelineBot` candle-trail writes dry-run order-intent rows with symbol,
    side, lot, SL/TP pips, confidence, magic, and run id.
  - `scripts/v5_reconcile_dry_run.py` compares expected V5 replay trades with
    dry-run journal rows and writes `dry_run_reconciliation.json`.
- [x] Add live-current dry-run watcher tooling.
  - `scripts/v5_watch_live_gate.py` waits for a journal row for the dedicated
    run id.
  - When the row exists, it calls strict live-current reconciliation without
    `--allow-mismatch`.
- [ ] Run dry-run first with a dedicated run id and artifact directory during
  open market hours.
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

### Dry-Run Gate Evidence

The first dry-run reconciliation command was executed against the existing
`data/live_trades.db` for `usdjpy-candle-trail-v5-enc30-hfm`. It produced
`data/v5_runs/usdjpy-candle-trail-v5-enc30-hfm/dry_run_reconciliation.json`.

Current result: `dry_run_mismatch`, `matched_orders=0/553`. This is expected
because the current journal contains old live/pending rows and no rows with the
new USDJPY HFM dry-run run id. The gate is implemented, but it has not passed.

A bounded 16-minute dry-run attempt loaded the USDJPY pipeline and candle models
successfully, but wrote `0` rows for `usdjpy-candle-trail-v5-enc30-hfm`. A short
unbuffered diagnostic run showed the blocker clearly: the bot cannot connect to
the local MT5 bridge on `localhost:18812` from this sandbox and reports:
`Cannot connect to mt5linux bridge ... [Errno 1] Operation not permitted`.

After starting `./start_mt5.sh`, the same dry-run command was rerun with
elevated local bridge access. One diagnostic run connected to account `54910316`
on `HFMarketsKE-Live2` and resolved `USDJPY` to `USDJPYr`. The next bounded
paper-gate run connected to demo account `57462638` on `HFMarketsKE-Demo2` and
resolved `USDJPY` to `USDJPY.Z`, which matches the primary `.Z` broker-profile
artifact.

Latest demo dry-run evidence:

- `2026-07-03 20:00:00`: `HOLD`, confidence `5.1%`;
- `2026-07-03 20:15:00`: `HOLD`, confidence `17.3%`.
- `2026-07-03 20:45:00`: `HOLD`, confidence `17.5%`;
- `2026-07-03 21:00:00`: `HOLD`, confidence `10.2%`;
- `2026-07-03 21:15:00`: `HOLD`, confidence `4.4%`.

No dry-run order-intent rows were written because all observed bars were holds, so the
current live dry-run validation reports `waiting_for_live_dry_run_order`.
The historical replay reconciliation still reports `dry_run_mismatch`,
`matched_orders=0/553`; this is expected because live July 2026 journal rows
cannot timestamp-match historical 2024-2026 replay entries.

Decision: bridge access works when run outside the sandbox restriction. For the
current demo account, keep using the `USDJPY.Z` HFM artifact and run id
`usdjpy-candle-trail-v5-enc30-hfm`. Keep dry-run running until an actionable
candle-trail signal occurs, then validate current live dry-run rows with
`live_dry_run_validation.json` before demo-live.

Weekend status on 2026-07-04: the FX market is closed, so no fresh actionable
USDJPY candle-trail BUY/SELL should be expected today. Do not count weekend
HOLD output or missing journal rows as a failed live gate. Resume this gate when
the market is open and the MT5 bridge is connected.

Preferred open-market watcher command:

```bash
conda run -n envmt5 --no-capture-output python scripts/v5_watch_live_gate.py \
  --run-dir data/v5_runs/usdjpy-candle-trail-v5-enc30-hfm \
  --journal data/live_trades.db \
  --run-id usdjpy-candle-trail-v5-enc30-hfm \
  --broker-symbol USDJPY.Z \
  --magic 20260103 \
  --expected-sl-pips 10 \
  --expected-tp-pips 30 \
  --max-lot 0.01 \
  --poll-seconds 30 \
  --max-polls 240
```

Dry-run command to produce journal rows for the primary candidate:

```bash
conda run -n envmt5 --no-capture-output python src/bots/pipeline_bot.py \
  --dry-run \
  --symbol USDJPY \
  --flip-mode candle_trail \
  --model-dir data/models/pipeline_USDJPY_v2 \
  --candle-model-dir data/models/candle_USDJPY \
  --magic 20260103 \
  --journal-run-id usdjpy-candle-trail-v5-enc30-hfm \
  --trail-activation-pips 15 \
  --trail-pips-behind 10 \
  --trail-max-bars-low 1 \
  --trail-max-bars-med 2 \
  --trail-max-bars-high 4 \
  --max-lot 0.01
```

Live-current validation command after dry-run rows exist:

```bash
conda run -n envmt5 --no-capture-output python scripts/v5_reconcile_dry_run.py \
  --live-current \
  --run-dir data/v5_runs/usdjpy-candle-trail-v5-enc30-hfm \
  --journal data/live_trades.db \
  --run-id usdjpy-candle-trail-v5-enc30-hfm \
  --broker-symbol USDJPY.Z \
  --magic 20260103 \
  --expected-sl-pips 10 \
  --expected-tp-pips 30 \
  --max-lot 0.01
```

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
conda run -n envmt5 python scripts/v5_validate_champion.py --mode hybrid-v2 --symbol EURUSD --data data/EURUSD_M15.csv --candle-features data/features/candle_signal_EURUSD.parquet --run-id eurusd-hybrid-v2-strict --train-days 180 --test-days 30 --model xgboost --max-folds 5
conda run -n envmt5 python scripts/v5_validate_champion.py --mode hybrid-v2 --symbol USDJPY --data data/USDJPY_M15.csv --candle-features data/features/candle_signal_USDJPY.parquet --run-id usdjpy-hybrid-v2-strict --train-days 180 --test-days 30 --model xgboost --max-folds 5
conda run -n envmt5 python scripts/v5_validate_champion.py --mode candle-trail --symbol EURUSD --data data/EURUSD_M15.csv --signals data/features/candle_signal_EURUSD.parquet --run-id eurusd-candle-trail-v5 --threshold 0.60 --sl-pips 10 --tp-pips 30
conda run -n envmt5 python scripts/v5_validate_champion.py --mode candle-trail --symbol USDJPY --data data/USDJPY_M15.csv --signals data/features/candle_signal_USDJPY.parquet --run-id usdjpy-candle-trail-v5 --threshold 0.60 --sl-pips 10 --tp-pips 30
conda run -n envmt5 python scripts/v5_build_candle_oos.py --symbol EURUSD --data data/EURUSD_M15.csv --out data/features/candle_signal_v5_EURUSD.parquet --train-days 120 --test-days 60
conda run -n envmt5 python scripts/v5_build_candle_oos.py --symbol USDJPY --data data/USDJPY_M15.csv --out data/features/candle_signal_v5_USDJPY.parquet --train-days 120 --test-days 60
```

Future heavier validation commands should be added here as they are built, with
the exact run id and output artifact path.
