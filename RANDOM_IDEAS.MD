Yes, please provide a detailed 90-day development roadmap
I need a project folder structure for a modular trading system
Can you list the exact tech stack and libraries for this MVP
What specific milestones should I target for the first phase
Please outline a 90-day plan focused on the initial data pipeline
Completed 2 steps
Absolutely — below is a practical 90-day roadmap designed to get you from zero to a stable initial data pipeline for a modular MT5 trading system. It is built around Python + MT5 + time-series-safe ML practices, with a feature store and experiment tracking ready for later scaling.

MVP stack
For the first phase, keep the stack tight and boring. Use Python, the MetaTrader5 Python package for broker connectivity, pandas and numpy for data handling, scikit-learn for baseline validation, XGBoost for tabular probability models, pandas-ta or pandas-ta-classic for indicators, Feast for feature management, and MLflow for experiment tracking.

For storage, start with SQLite or Parquet files locally, then move to PostgreSQL or TimescaleDB once the pipeline is stable. For orchestration and API boundaries, add FastAPI and Docker early so the architecture stays modular.

Exact libraries
MetaTrader5

pandas

numpy

scikit-learn

xgboost

pandas-ta or pandas-ta-classic

mlflow

feast

pyarrow

sqlalchemy

fastapi

uvicorn

pydantic

python-dotenv

loguru

pytest

Project structure
Use a structure that separates raw data, features, models, execution, and orchestration. That way each piece can be swapped later without rewriting the whole system.

text
trading_platform/
├── README.md
├── .env
├── pyproject.toml
├── docker/
├── configs/
│   ├── app.yaml
│   ├── mt5.yaml
│   ├── features.yaml
│   ├── train.yaml
│   └── backtest.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── features/
│   └── samples/
├── docs/
├── logs/
├── notebooks/
├── reports/
├── models/
│   ├── registry/
│   └── artifacts/
├── src/
│   └── trading_platform/
│       ├── __init__.py
│       ├── main.py
│       ├── config/
│       ├── core/
│       │   ├── constants.py
│       │   ├── types.py
│       │   └── exceptions.py
│       ├── data/
│       │   ├── mt5_client.py
│       │   ├── collectors.py
│       │   ├── schemas.py
│       │   └── validators.py
│       ├── features/
│       │   ├── indicators.py
│       │   ├── regimes.py
│       │   ├── transforms.py
│       │   └── feature_pipeline.py
│       ├── signals/
│       │   ├── rules.py
│       │   └── signal_base.py
│       ├── models/
│       │   ├── base.py
│       │   ├── xgb_model.py
│       │   ├── calibration.py
│       │   └── ensemble.py
│       ├── backtest/
│       │   ├── engine.py
│       │   ├── metrics.py
│       │   └── cost_model.py
│       ├── execution/
│       │   ├── order_manager.py
│       │   ├── risk_manager.py
│       │   └── broker_adapter.py
│       ├── pipeline/
│       │   ├── ingest.py
│       │   ├── build_features.py
│       │   ├── train.py
│       │   └── infer.py
│       ├── monitoring/
│       │   ├── logging.py
│       │   ├── drift.py
│       │   └── metrics.py
│       └── api/
│           ├── app.py
│           └── routes.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
└── scripts/
    ├── download_history.py
    ├── build_dataset.py
    ├── train_baseline.py
    └── run_backtest.py
This layout gives you clean boundaries for data, features, models, and execution. It also makes it easy to convert parts into services later.

90-day roadmap
The first 90 days should not try to solve trading profitability. The goal is a reliable data pipeline that produces clean, versioned, time-aligned datasets ready for future modeling.

Days 1-15: Foundation
Set up the repository, virtual environment, dependencies, config management, logging, and folder structure. Create the MT5 connection module and verify you can fetch account info, symbols, bars, and ticks reliably from the terminal.

Milestones:

Repo initialized.

.env and config system working.

MT5 connection tested.

Historical candle download script working.

Raw data saved in Parquet or CSV.

Days 16-30: Data ingestion
Build repeatable ingestion for one symbol and one timeframe first, such as EURUSD M5. Save raw candles and ticks with timestamps, source metadata, and ingestion logs. Add validation so missing bars, duplicate timestamps, and bad OHLC values are caught early.
Milestones:

One-symbol historical downloader complete.

Incremental update job complete.

Raw schema validator added.

Data quality checks for gaps and duplicates.

Data saved consistently in data/raw/.

Days 31-45: Normalization and storage
Create a standardized intermediate dataset with cleaned timestamps, timezone handling, symbol naming, and bar alignment. Convert raw data into data/interim/ and data/processed/ outputs with reproducible scripts.
Milestones:

Timezone normalization done.

Resampling/alignment rules defined.

Processed candle dataset reproducible.

Database or Parquet storage convention finalized.

Data lineage fields added.

Days 46-60: Feature pipeline
Implement your first feature set using simple indicators and market structure features. Start with returns, rolling volatility, moving averages, RSI, MACD, ATR, candle body/wick ratios, and session/time-of-day features using pandas-ta or pandas-ta-classic.

Milestones:

Feature functions written as reusable modules.

Feature generation script working.

Feature output saved to data/features/.

Feature versioning introduced.

NaN handling and lookback-window rules tested.

Days 61-75: Dataset assembly
Build a labeled dataset from features and forward-looking targets. Use time-series-safe splitting with TimeSeriesSplit, because standard random cross-validation leaks future information.

Milestones:

Labeling logic created.

Train/validation/test split by time.

Leakage checks completed.

Baseline dataset exported.

MLflow experiment tracking connected.

Days 76-90: First baseline pipeline
Train a simple baseline classifier or regressor, preferably XGBoost or logistic regression, using time-ordered data. Focus on probability outputs and calibration rather than direct predictions; your objective is to verify the pipeline works end to end, not to maximize returns yet.

Milestones:

Baseline model trained.

Probability calibration evaluated.

Model artifacts logged.

Reproducible train script created.

Pipeline can rebuild dataset and rerun training from scratch.

First-phase milestones
Your first phase should be complete when all of these are true:

You can download MT5 data on demand and incrementally.

Raw, interim, and processed datasets are reproducible.

Feature generation is modular and versioned.

Time-series splits are leakage-safe.

At least one baseline model can train from the pipeline.

Model artifacts and metrics are tracked in MLflow.

The pipeline can run from config, not manual notebook steps.

Data pipeline focus
For the initial pipeline, the most important design choice is to build everything around idempotent jobs. Each script should be safe to rerun and produce the same result given the same input. That reduces debugging pain and makes later automation much easier.
You should also keep raw data immutable, store transformations separately, and never mix feature generation with execution logic. A feature store like Feast becomes useful later once the offline/online consistency problem starts to matter.

Recommended first build order
MT5 connector and history downloader.

Raw data storage and schema validation.

Cleaning, resampling, and timezone normalization.

Feature generation.

Time-based labeling and splitting.

MLflow experiment logging.

Baseline probability model.

Simple backtest on historical signals.

What to avoid
Do not start with LLMs, reinforcement learning, or a full microservices setup. Do not try to trade many symbols or many timeframes at once. Do not use random train-test splits for market data, because they leak future information and create misleading results.

The first 90 days should produce a dependable research foundation, not a finished money-making system.

Would you like me to turn this into a week-by-week checklist with deliverables, or draft the actual starter pyproject.toml and module skeleton next?


