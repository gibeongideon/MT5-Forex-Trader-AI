import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.feature_pipeline import FeaturePipeline
from src.v5.validation import (
    BrokerExecutionRules,
    V5ValidationError,
    assert_candle_predictions_are_oos,
    assert_feature_builder_is_past_only,
    assert_fold_fit_records_are_train_only,
    assert_labels_match_forward_returns,
)


def _ohlcv(n=90):
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    close = 1.1000 + np.sin(np.arange(n) / 8) * 0.002 + np.arange(n) * 0.00001
    return pd.DataFrame(
        {
            "open": close - 0.00005,
            "high": close + 0.0002,
            "low": close - 0.0002,
            "close": close,
            "tick_volume": np.arange(n) + 100,
        },
        index=idx,
    )


def test_feature_builder_rejects_future_dependent_features():
    df = _ohlcv()

    def leaky_builder(frame):
        feature = frame["close"].shift(-1).rename("future_close")
        return pd.DataFrame({"future_close": feature}).dropna()

    with pytest.raises(V5ValidationError, match="future rows changed"):
        assert_feature_builder_is_past_only(
            df,
            leaky_builder,
            cutoff=df.index[50],
        )


def test_feature_pipeline_passes_past_only_guard():
    df = _ohlcv()
    pipe = FeaturePipeline(scale=False)

    def build(frame):
        X, _ = pipe.build(frame, fit=True)
        return X

    report = assert_feature_builder_is_past_only(df, build, cutoff=df.index[70])

    assert report["checked_rows"] > 0
    assert report["checked_columns"] >= 20


def test_labels_must_match_configured_forward_returns():
    df = _ohlcv()
    close = df["close"]
    horizon = 3
    threshold = 0.0002
    future_return = close.shift(-horizon) / close - 1
    labels = pd.Series(0, index=close.index)
    labels[future_return > threshold] = 1
    labels[future_return < -threshold] = -1
    labels = labels.iloc[:-horizon]

    report = assert_labels_match_forward_returns(
        close,
        labels,
        horizon=horizon,
        threshold=threshold,
    )

    assert report["checked_labels"] == len(labels)


def test_label_guard_rejects_shifted_label_bug():
    df = _ohlcv()
    bad_labels = pd.Series(0, index=df.index[:-2])

    with pytest.raises(V5ValidationError, match="label mismatches"):
        assert_labels_match_forward_returns(
            df["close"],
            bad_labels,
            horizon=2,
            threshold=0.0001,
        )


def test_fold_fit_records_reject_transformers_fit_beyond_train_window():
    train_end = pd.Timestamp("2026-02-01")
    records = [
        {
            "fold": 0,
            "component": "encoder",
            "train_start": pd.Timestamp("2026-01-01"),
            "train_end": train_end,
            "fit_start": pd.Timestamp("2026-01-01"),
            "fit_end": pd.Timestamp("2026-02-15"),
        }
    ]

    with pytest.raises(V5ValidationError, match="fit_end"):
        assert_fold_fit_records_are_train_only(records)


def test_candle_predictions_must_be_fold_oos_and_unique():
    preds = pd.DataFrame(
        {
            "fold": [0, 0, 1],
            "prediction_time": pd.to_datetime(
                ["2026-02-02", "2026-02-03", "2026-03-02"]
            ),
            "train_start": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-01"]
            ),
            "train_end": pd.to_datetime(
                ["2026-02-01", "2026-02-01", "2026-03-01"]
            ),
            "test_start": pd.to_datetime(
                ["2026-02-01", "2026-02-01", "2026-03-01"]
            ),
            "test_end": pd.to_datetime(
                ["2026-03-01", "2026-03-01", "2026-04-01"]
            ),
        }
    )

    report = assert_candle_predictions_are_oos(preds)

    assert report["checked_predictions"] == 3


def test_candle_oos_guard_rejects_train_window_predictions():
    preds = pd.DataFrame(
        {
            "fold": [0],
            "prediction_time": pd.to_datetime(["2026-01-15"]),
            "train_start": pd.to_datetime(["2026-01-01"]),
            "train_end": pd.to_datetime(["2026-02-01"]),
            "test_start": pd.to_datetime(["2026-02-01"]),
            "test_end": pd.to_datetime(["2026-03-01"]),
        }
    )

    with pytest.raises(V5ValidationError, match="outside OOS test window"):
        assert_candle_predictions_are_oos(preds)


def test_broker_execution_rules_round_and_cap_lots():
    rules = BrokerExecutionRules(
        pip_size=0.0001,
        spread_pips=1.2,
        commission_pips=0.5,
        slippage_pips=0.3,
        entry_delay_bars=1,
        min_lot=0.01,
        lot_step=0.01,
        max_lot=0.50,
    )

    assert rules.normalize_lot(0.004) == 0.0
    assert rules.normalize_lot(0.026) == 0.02
    assert rules.normalize_lot(0.876) == 0.50
    assert rules.round_trip_cost_pips == pytest.approx(2.0)
