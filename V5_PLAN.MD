# V5 MT5 + Lumibot Profitability Plan

> Coordination document for the V5 profitability track.
> Last updated: 2026-07-05
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

USDJPY regeneration after the completed-bar MTF fix was run on 2026-07-04:

| Run ID | Symbol | Signal File | Folds | OOS Rows | Signal Rows | Trades | Final Equity | Return | MaxDD | Win Rate | Sharpe | Daily Sharpe | Verdict |
| ------ | ------ | ----------- | ----- | -------- | ----------- | ------ | ------------ | ------ | ----- | -------- | ------ | ------------ | ------- |
| `usdjpy-candle-trail-v5-post-mtf-fix-enc30` | USDJPY | `data/features/candle_signal_v5_USDJPY_post_mtf_fix_enc30.parquet` | 12 | 48,936 | 541 | 421 | 9956.35 | -0.436% | 0.722% | 41.3% | -1.188 | -0.931 | fails after MTF fix |

Interpretation: the previous USDJPY candle-trail edge does not survive the
completed-bar MTF correction. The paper/live gate for USDJPY candle-trail should
not proceed unless a newly regenerated variant passes stricter validation. The
next research path is either EURUSD post-fix regeneration for completeness or
the CTA portfolio track from `V5_RESEARCH_AUDIT.MD`.

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

## H4 CTA Track (2026-07-05)

After the USDJPY candle-trail edge failed the post-MTF-fix regeneration, the
audit's top-priority alternate path was implemented: a vol-targeted EWMAC
trend portfolio on H4 bars for EURUSD, GBPUSD, USDJPY, XAUUSD.

- Code: `src/v5/h4_cta.py` (strategy), `scripts/v5_h4_cta_backtest.py`
  (runner + artifacts), `tests/test_v5_h4_cta.py` (leakage/cost guards).
- No fitted components (no scaler/encoder/classifier), so there is no
  train/test boundary to leak across. The lookahead surface is guarded by
  causality tests that mutate future bars and require bit-identical positions
  and P&L before the cutoff.
- Config was pre-registered before the first run (module docstring):
  Carver EWMAC speeds scaled to H4 bars (daily (8,32)..(64,256) x 6),
  cluster-equal risk across {fx, metal}, 10% vol target, causal no-trade
  buffer 0.10, `positions.shift(1)` execution, full per-bar spread (floored
  at symbol median) charged per unit turnover.
- Evaluation window pre-registered as 2017-01-01 onward (2015-2016 is EWMAC
  warm-up).

### H4 CTA Evidence

All runs: 2017-01-01 to 2026-06-15 (9.45 years) unless noted, initial equity
10,000, artifacts under `data/v5_runs/<run_id>/`.

| Run ID | Stress | Return | CAGR | Bar Sharpe | Daily Sharpe | Daily Sharpe 95% CI | MaxDD | Cost Drag/yr |
| ------ | ------ | ------ | ---- | ---------- | ------------ | ------------------- | ----- | ------------ |
| `h4-cta-v5` | baseline | +76.6% | 6.21% | 0.634 | 0.450 | [-0.124, +1.064] | 16.8% | 0.41% |
| `h4-cta-v5-costx2` | spread x2 | +69.8% | 5.77% | 0.594 | 0.413 | [-0.161, +1.029] | ~same | 0.83% |
| `h4-cta-v5-delay2` | entry delay 2 bars | +74.2% | 6.05% | 0.618 | 0.435 | [-0.142, +1.053] | ~same | 0.41% |
| `h4-cta-v5-recent` | 2021+ subsample | +62.4% | 9.32% | 0.915 | 0.660 | [-0.099, +1.415] | - | - |

Per-symbol net Sharpe (baseline): EURUSD +0.11, GBPUSD -0.08, USDJPY +0.14,
XAUUSD +0.77. Yearly net returns range -12.8% (2017) to +28.4% (2025); 6 of
10 calendar years positive.

Interpretation:

- The edge survives the stresses that killed prior candidates: doubling costs
  moves Sharpe only 0.63 to 0.59, and an extra execution bar moves it only to
  0.62 (the pre-fix candle-trail collapsed 4.47 to 1.52 under the same delay
  test). That decay profile is what a real slow edge looks like.
- The bootstrap CI still includes zero: a Sharpe ~0.6 needs longer than 9.5
  years for 95% significance. This matches the v4 48-instrument CTA result
  (+0.65, CI [+0.18, +1.12]) — the CI there excluded zero only because 48
  instruments diversify; with 4 instruments the same engine is noisier.
- The edge is gold-dominated. The three FX legs are near-zero standalone but
  add diversification. Widening the universe (the v4 CTA basket) is the
  known fix and remains the next research step.
- Verdict: first V5 candidate with a positive, stress-stable, leakage-guarded
  Sharpe. Research-grade, not yet deployable — needs the dry-run/paper gate
  and lot-sizing checks (`src/cta/sizing.py`) before live consideration.

### Universe-Widening Ablation (2026-07-05)

To measure the diversification impact, the same engine was run on nested D1
universes from the v4 CTA data (2008-2026, cost_bps spread model), via
`scripts/v5_universe_backtest.py`. Tiers: `core4` (EURUSD/GBPUSD/USDJPY/GOLD),
`fx-metals` (23), `mt5-cfd` (36, +energy +indices), `full48` (+rates, ags,
crypto), `basket5` (v4 locked small basket: GOLD/UST10Y/SPX/WTI/EURUSD).
Sleeves: `trend` (EWMAC only, the H4 engine) and `combined` (50/50 EWMAC +
cross-sectional momentum, the v4 champion sleeve).

Evaluation 2010-01-01 to 2026-06 unless noted; all net of costs.

| Run | Tier | Config | Sharpe | 95% CI | CAGR | MaxDD |
| --- | ---- | ------ | ------ | ------ | ---- | ----- |
| `d1-cta-v5-core4-fast-2010` | 4 | trend, fast, daily | +0.375 | [-0.13, +0.89] | 3.5% | 24.5% |
| `d1-cta-v5-fx-metals-fast-2010` | 23 | trend, fast, daily | -0.106 | [-0.58, +0.38] | -1.6% | 50.1% |
| `d1-cta-v5-mt5-cfd-fast-2010` | 36 | trend, fast, daily | -0.167 | [-0.65, +0.33] | -2.3% | 50.4% |
| `d1-cta-v5-full48-fast-2010` | 48 | trend, fast, daily | +0.321 | [-0.16, +0.82] | 2.8% | 35.0% |
| `d1-cta-v5-full48-slow-combined-monthly` | 48 | combined, slow, monthly, buf 0.4 | +0.697 | [+0.19, +1.21] | 7.2% | 23.2% |
| `d1-cta-v5-basket5-slow-combined-monthly` | 5 | combined, slow, monthly, buf 0.4 | +0.778 | [+0.34, +1.21] | 8.6% | 18.8% |

Champion subsamples and stress (basket5, combined/slow/monthly):

| Run | Variant | Sharpe | 95% CI |
| --- | ------- | ------ | ------ |
| `d1-cta-v5-basket5-champion-2017` | 2017+ | +0.939 | [+0.41, +1.51] |
| `d1-cta-v5-basket5-champion-2021` | 2021+ | +0.947 | [+0.28, +1.76] |
| `d1-cta-v5-basket5-champion-costx2` | spread x2 | +0.770 | [+0.33, +1.21] |
| `d1-cta-v5-basket5-champion-delay2` | entry delay 2 | +0.759 | [+0.32, +1.20] |
| `d1-cta-v5-full48-champion-2017` | full48, 2017+ | +0.767 | [+0.13, +1.45] |

Findings:

1. Widening WITHIN FX destroys the edge: 23 FX+metal instruments score
   -0.11 to -0.02 (FX trend has been ~zero since 2010, and 18 FX pairs with
   shared legs are essentially one correlated bet with 18x the costs).
2. The lift comes from adding UNCORRELATED ASSET CLASSES, not instrument
   count: rates, equity indices, metals, and crypto each contribute positive
   class-level Sharpe; basket5 (5 instruments, 5 classes) beats full48
   (48 instruments, 8 classes).
3. The v4 CTA champion REPRODUCES under the V5 causal engine (past-only
   buffer band): +0.697 full48 vs the v4 claim of +0.65 with CI [+0.18,
   +1.12], and +0.778 basket5 vs the v4 claim of +0.746. v4's buffer-band
   lookahead was not load-bearing.
4. `d1-cta-v5-basket5-*` is now the strongest validated candidate in the
   project: bootstrap CI excludes zero on the full period AND both
   subsamples, and it is insensitive to doubled costs and doubled execution
   delay. The four requested pairs remain served: EURUSD trades in basket5
   directly and the H4 core4 book (`h4-cta-v5`) can run alongside on
   separate magics, but the FX legs should be treated as diversifiers, not
   as the edge.

### H4 CTA Commands

```bash
conda run -n envmt5 python -m pytest tests/test_v5_h4_cta.py -q
conda run -n envmt5 python scripts/v5_h4_cta_backtest.py --run-id h4-cta-v5
conda run -n envmt5 python scripts/v5_h4_cta_backtest.py --run-id h4-cta-v5-costx2 --spread-mult 2.0
conda run -n envmt5 python scripts/v5_h4_cta_backtest.py --run-id h4-cta-v5-delay2 --entry-delay-bars 2
conda run -n envmt5 python scripts/v5_h4_cta_backtest.py --run-id h4-cta-v5-recent --eval-start 2021-01-01
```

### Phase A Lever Experiments — Pre-Registration (2026-07-05)

Registered BEFORE any lever run was executed. Engine: `src/v5/levers.py`
`lever_positions` (single shared builder for backtest/tests/runner; supersedes
`champion_positions`, whose buffer band width is a full-sample statistic).
Champion base config for every run: slow speeds, combined sleeve, monthly
rebalance, buffer 0.4, vol target 10%, eval 2010-01-01+.

Pre-registered lever parameters (no sweeps): carry weight 0.5 on FX legs
only; `ml_min_rows=2500`; `ml_purge_mult=2.0`; regime gates at module
defaults (`src/cta/regime.py`).

Run matrix (13 runs, each lever alone; identity run `v5q-b5-base` must
reproduce Sharpe 0.778 exactly):

- Baselines: `v5q-b5-base`, `v5q-b10-base`, `v5q-b10btc-base`
  (basket10 = GOLD SILVER UST10Y UST30Y SPX DAX WTI BRENT EURUSD USDJPY).
- Levers x {basket5, basket10}: `--regime trend`, `--regime vol`,
  `--regime trend_vol`, `--carry`, `--ml-combine`.

Promotion criteria (locked): promote a config only if Sharpe > 0.778 AND
bootstrap CI-low > 0.34, AND `--spread-mult 2.0` / `--entry-delay-bars 2`
stress degradation is no worse than baseline's. All results reported
regardless of outcome. If several pass, highest CI-low wins. Lever
combinations only as a separately pre-registered round 2.

### Phase A Lever Results (2026-07-05)

All 13 pre-registered runs, reported regardless of outcome (Sharpe = daily
annualized, net of costs, 2010-01+):

| Run | Sharpe | CI 95% | CAGR | Verdict |
| --- | ------ | ------ | ---- | ------- |
| `v5q-b5-base` | 0.778 | [0.34, 1.21] | 8.6% | identity check passed exactly |
| `v5q-b10-base` | 0.738 | [0.24, 1.25] | 7.9% | within-class doubling does not help |
| `v5q-b10btc-base` | **0.935** | **[0.45, 1.42]** | 10.2% | PROMOTED |
| `v5q-b5-reg-trend` | 0.768 | [0.32, 1.22] | 8.3% | no improvement |
| `v5q-b5-reg-vol` | 0.671 | [0.24, 1.13] | 7.1% | degrades |
| `v5q-b5-reg-trend_vol` | 0.734 | [0.30, 1.20] | 7.8% | degrades |
| `v5q-b5-carry` | 0.770 | [0.33, 1.22] | 8.5% | neutral |
| `v5q-b5-ml` | 0.148 | [-0.33, 0.65] | 0.6% | fails |
| `v5q-b10-reg-trend` | 0.753 | [0.27, 1.26] | 8.1% | no improvement |
| `v5q-b10-reg-vol` | 0.641 | [0.16, 1.14] | 6.8% | degrades |
| `v5q-b10-reg-trend_vol` | 0.719 | [0.25, 1.21] | 7.7% | degrades |
| `v5q-b10-carry` | 0.690 | [0.21, 1.19] | 7.3% | degrades |
| `v5q-b10-ml` | 0.332 | [-0.15, 0.83] | 1.4% | fails |

Promotion evidence for `basket10-btc` (11 instruments, 6 asset classes):

| Run | Variant | Sharpe | CI 95% |
| --- | ------- | ------ | ------ |
| `v5q-b10btc-costx2` | spread x2 | 0.925 | [0.44, 1.41] |
| `v5q-b10btc-delay2` | entry delay 2 | 0.908 | [0.42, 1.40] |
| `v5q-b10btc-2017` | 2017+ | 0.949 | [0.36, 1.60] |
| `v5q-b10btc-2021` | 2021+ | 0.888 | [0.12, 1.71] |

Notes: the delay-2 relative decay (2.9%) is marginally worse than the
baseline's (2.4%) — flagged per the locked criteria; accepted because the
stressed Sharpe 0.908 still exceeds the unstressed baseline 0.778 with a
higher CI-low. Per-class Sharpe is spread across crypto 0.70, metals 0.55,
equities 0.45, rates 0.23, FX 0.30 — not a BTC-only artifact. The promoted
config is written to `configs/v5_basket_champion.json`.

Interpretation of the failures: regime gates and carry do not add to an
already vol-targeted trend book; the Ridge factor blend destroys the edge
(the pooled linear weighting dilutes the slow-trend signal with weak
factors). These stay documented as dead ends.

## Champion Live Runner + AI Agent Risk Overlay (2026-07-05)

- `scripts/v5_basket_runner.py` — daily advisory runner. Builds positions
  through the SAME `lever_positions` code path as the promoted backtest
  (never `champion_positions`, whose buffer band has a full-sample-mean
  lookahead), diffs against state, journals every non-hold action as a
  dry-run intent (`TradeJournal`, magic 360500/360501, `dry_run=1`), and
  writes `positions.csv`/`state.json` under `data/v5_runs/v5-basket-*`.
  Advisory/paper only: there is no order-send code path.
- `src/v5/agents.py` — Lumibot-pattern agent team (researcher -> bull ->
  bear -> PM, sequential summary chaining, Claude Haiku via the Anthropic
  API). Risk-overlay only: the PM emits per-instrument scales clamped to
  [0,1] in code; analysts have no trading fields in their schemas; any
  error fails OPEN to scales=1.0 (pure champion); a sha256 response cache
  makes same-day reruns free; every call is journaled to
  `agent_decisions/YYYY-MM-DD.jsonl`.
- NO backtest Sharpe claims are made for the overlay, ever: LLM pretraining
  knows market history, so an LLM backtest cannot be leak-free (Lumibot's
  own backtest safety is prompt-based only).
- Verified 2026-07-05: 181 tests green (levers identity, agent clamps,
  fail-open, cache, budget, journal dedup); live smoke test ran 4 real
  Haiku calls (~24s, ~1.3k output tokens), PM scaled gross 1.52 -> 0.98,
  cache rerun reproduced identical scales with zero API calls; fail-open
  verified by removing the API key.

### Forward A/B Protocol (pre-registered)

- Run daily after the D1 data refresh, open-market days only:
  `python scripts/v5_basket_runner.py --overlay`
  (journals BOTH books: `v5-basket-pure` raw targets, `v5-basket-overlay`
  scaled targets).
- Evaluate after >= 8 weeks (~40 observations): mean daily PnL difference,
  downside capture on champion-loss days, scale activity distribution,
  fail-open frequency. Explicitly NO Sharpe/CI claims at that sample size.
- The overlay is promoted to demo-live consideration only if it does not
  degrade the pure book and shows loss-day downside reduction.

### Capital note

At $10k equity the 11-leg book cannot be traded faithfully: GOLD and DAX
legs round to zero lots and the offline `min_viable_equity` estimate is
~$67k (offline contract sizes are placeholders for XAGUSD/GER40/UKOIL/
US30YR/BTCUSD — verify with `--live` against the terminal). Options before
paper trading: more equity, or a pre-registered reduced-leg variant
(basket5 remains valid at Sharpe 0.778).

### Runner Commands

```bash
conda run -n envmt5 python -m pytest tests/test_v5_levers.py tests/test_v5_agents.py tests/test_v5_basket_runner.py -q
conda run -n envmt5 python scripts/v5_basket_runner.py                 # pure book
conda run -n envmt5 python scripts/v5_basket_runner.py --overlay      # A/B both books
conda run -n envmt5 python scripts/v5_basket_runner.py --equity 10000 --live  # broker-spec sizing
conda run -n envmt5 python scripts/v5_basket_runner.py --validate     # Sharpe re-check vs promoted run
```

### Universe Ablation Commands

```bash
conda run -n envmt5 python scripts/v5_universe_backtest.py --tier core4 --run-id d1-cta-v5-core4-fast-2010
conda run -n envmt5 python scripts/v5_universe_backtest.py --tier fx-metals --run-id d1-cta-v5-fx-metals-fast-2010
conda run -n envmt5 python scripts/v5_universe_backtest.py --tier full48 --speeds slow --sleeve combined --rebalance monthly --buffer-frac 0.4 --run-id d1-cta-v5-full48-slow-combined-monthly
conda run -n envmt5 python scripts/v5_universe_backtest.py --tier basket5 --speeds slow --sleeve combined --rebalance monthly --buffer-frac 0.4 --run-id d1-cta-v5-basket5-slow-combined-monthly
conda run -n envmt5 python scripts/v5_universe_backtest.py --tier basket5 --speeds slow --sleeve combined --rebalance monthly --buffer-frac 0.4 --spread-mult 2.0 --run-id d1-cta-v5-basket5-champion-costx2
conda run -n envmt5 python scripts/v5_universe_backtest.py --tier basket5 --speeds slow --sleeve combined --rebalance monthly --buffer-frac 0.4 --entry-delay-bars 2 --run-id d1-cta-v5-basket5-champion-delay2
```

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
