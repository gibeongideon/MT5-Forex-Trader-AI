# Rules for Building This App

> This document is for any developer or AI agent starting a new session on this project.
> Follow these rules every time — they prevent regressions, broken backtests, and wasted rebuilding.

---

## Session Start Protocol

Run these steps **in order** before writing any code:

1. **Read the plan**
   Open `IMPLEMENTATION-PLAN.MD` and read it in full.
   Understand the 10-phase architecture and which phase you are working on.

2. **Check progress**
   Open `PROGRESS.md`.
   Find the current phase status table and the "Next task" line.
   Do not re-implement anything already marked ✅ COMPLETE.

3. **Read context if needed**
   Open `GENERAL_FILES.MD` for a plain-English explanation of what the backtest measures,
   what SL/TP means, and what metrics are accurate vs. simplified.

4. **Check the environment**
   Open `config.yaml` to see the current bot/model configuration.

5. **Then write code**
   Implement only what the current phase requires.
   Do not jump phases or add features not in the plan.

---

## Architecture Rules (Never Break These)

These rules are the foundation of the entire system.
Violating any one of them breaks the plug-in model architecture.

| # | Rule |
|---|------|
| 1 | Every signal source must output `[P_buy, P_hold, P_sell]` — a probability triple that sums to 1.0. Never return hard buy/sell/hold strings from the signal layer. |
| 2 | `BotBase` (`src/bot_base.py`) and `MT5Connector` (`src/mt5_connector.py`) interfaces are **frozen**. Extend them, never rewrite them. |
| 3 | Every model must implement `ModelInterface` from `src/model_interface.py`. The trading engine calls `predict_proba(X)` — it does not care what model is underneath. |
| 4 | **No lookahead in features.** All rolling computations must use `shift(1)` or older data. The feature pipeline validates this explicitly — run `validate_no_lookahead()` when adding new features. |
| 5 | The confidence threshold gates trades. Never bypass it. Default is 0.55 (55%). Below threshold → hold. Configured per bot in `config.yaml`. |
| 6 | Models are swappable via `config.yaml: active_model`. Switching models requires zero code changes. |
| 7 | Any model's probability output can become an input feature to another model. This is the stacking pattern (Phase 6). |

---

## Code Quality Rules

- **No hard-coded paths.** Use `config.yaml` values or argparse `default=` arguments.
- **Scripts must be idempotent.** Safe to re-run — same inputs produce same outputs.
- **Raw data is immutable.** Never modify or overwrite `data/EURUSD_M15.csv` or anything in `data/raw/`.
- **Scaler fitted on training data only.** Call `pipeline.build(df, fit=True)` for training. Call `pipeline.build(df, fit=False)` for test/live. Never re-fit on test data.
- **XGBoost label encoding.** XGBoost requires 0-indexed classes. Labels -1/0/1 must be remapped to 0/1/2 using `np.searchsorted(classes, y)` before `.fit()`, then output reordered back to `[P_buy, P_hold, P_sell]` after `predict_proba()`.
- **Use `Optional[dict]` not default mutable args.** Never use `def f(data={})` — always `def f(data=None)`.

---

## Environment Rules

| Rule | Detail |
|------|--------|
| Always activate the conda env | `conda activate envmt5` or prefix with `conda run -n envmt5 python ...` |
| Python version | 3.10 (in conda env `envmt5`) |
| Pinned — do NOT upgrade | `numpy==1.26.4`, `pandas==2.2.3`, `scipy==1.11.4`, `scikit-learn==1.1.1` |
| Safe install method | `conda install -n envmt5 -c conda-forge <package>` |
| If conda install fails | Use the env's own pip: `/home/rock/anaconda3/envs/envmt5/bin/pip install <pkg>` |
| MT5 bridge | Only needed for live bots. rpyc on `localhost:18812`. Not needed for backtest or training scripts. |

---

## Project Structure Rules

```
MT5/
├── IMPLEMENTATION-PLAN.MD   ← 10-phase plan — read first every session
├── PROGRESS.md              ← session handoff — read second every session
├── GENERAL_FILES.MD         ← plain-English trading/backtest context
├── RULES_OF_BUILDING_THIS_APP.MD  ← this file
├── RANDOM_IDEAS.MD          ← reference ideas (already reviewed and incorporated)
├── config.yaml              ← all config — add sections per phase, never remove old ones
├── requirements.txt         ← pinned dependencies
│
├── src/
│   ├── core/                ← shared types, exceptions, constants (no dependencies)
│   ├── data/                ← MT5 client, data schemas, validators
│   ├── features/            ← indicator library + feature pipeline
│   ├── signals/             ← rule engine, signal combiners
│   ├── models/              ← all model implementations (xgboost, lightgbm, etc.)
│   ├── execution/           ← risk manager, order manager (Phase 8)
│   ├── monitoring/          ← drift detection, alerter (Phase 10)
│   ├── api/                 ← FastAPI dashboard (Phase 10)
│   │
│   ├── bot_base.py          ← FROZEN — all bots extend this
│   ├── mt5_connector.py     ← FROZEN — all MT5 I/O goes through here
│   ├── model_interface.py   ← FROZEN — all models implement this
│   ├── trade_journal.py     ← SQLite logger (Phase 1)
│   ├── metrics.py           ← performance functions (Phase 1)
│   ├── random_bot.py        ← random baseline bot (Phase 1)
│   ├── rule_bot.py          ← rule-based bot (Phase 2)
│   └── ai_bot.py            ← will be refactored in Phase 9
│
├── scripts/                 ← one-shot CLI tools (train, backtest, download)
│
├── data/
│   ├── raw/                 ← original downloaded CSVs (immutable, never edit)
│   ├── features/            ← computed feature matrices (Parquet)
│   ├── models/              ← saved model artifacts (.joblib)
│   └── trades.db            ← SQLite trade journal
│
└── tests/
    ├── unit/                ← per-module tests
    ├── integration/         ← multi-module flow tests
    └── e2e/                 ← end-to-end backtest tests
```

**Layer dependency order** (lower layers must not import from higher):
```
core → data → features → signals → models → execution → bot_base → bots
                                                       ↑
                                              metrics, trade_journal (standalone)
```

---

## Session End Protocol

After completing any work:

1. **Update `PROGRESS.md`:**
   - Change the phase status (⬜ NOT STARTED → 🔄 IN PROGRESS → ✅ COMPLETE)
   - Update "Next task"
   - Add new files to the file table with a one-line description
   - Note any new architecture decisions

2. **Commit with a clear message:**
   ```
   git commit -m "Phase N: <short description of what was built>"
   ```

3. **Do not commit:**
   - `data/trades.db` (SQLite — runtime artifact)
   - `data/models/*.joblib` (large binary — add to .gitignore if not already)
   - `data/features/*.parquet` (large — add to .gitignore if not already)
   - `.env` files or any file containing API keys

---

## Quick Reference: Key Patterns

### Adding a new model
1. Create `src/models/mymodel_model.py` implementing `ModelInterface`
2. Register it in `src/model_registry.py` (Phase 5+)
3. Add config key in `config.yaml` under `models:`
4. Run `scripts/walk_forward.py` to verify out-of-sample Sharpe

### Adding a new feature
1. Add the computation to `src/features/indicators.py` or `src/features/feature_pipeline.py`
2. Ensure the result is shifted by 1 bar before being added to the feature matrix
3. Run `validate_no_lookahead()` to confirm there is no data leakage
4. Re-run `scripts/build_features.py` to regenerate the Parquet feature matrix

### Adding a new bot
1. Create `src/mybot.py` extending `BotBase`
2. Override `on_tick()` — call `registry.get_active().predict_proba(features)` for the signal
3. Add a config section in `config.yaml`
4. Test with `--backtest` flag before connecting to MT5

---

## What Has Already Been Built

See `PROGRESS.md` for the current status table and a description of every file built so far.
The short version:

| Phase | What was built |
|-------|---------------|
| 1 | Random bot baseline, SQLite trade journal, performance metrics |
| 2 | Indicator library (SMA/EMA/RSI/MACD/BB/ATR/Stoch/ADX), rule engine, rule bot |
| 3 | Feature pipeline (31 features, no-lookahead validator, Parquet output) |
| 4 | XGBoost model, calibrated probabilities, walk-forward validator (Sharpe 1.34 out-of-sample) |
| 5–10 | Not yet started |
