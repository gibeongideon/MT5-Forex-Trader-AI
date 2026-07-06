import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.v5.candle_oos import (
    V5CandleOOSConfig,
    add_candle_extra_features,
    generate_candle_oos_predictions,
)
from src.v5.validation import assert_candle_predictions_are_oos


class RecordingCandleModel:
    def __init__(self):
        self.train_index = None
        self.feature_columns = None

    def train(self, X, y):
        self.train_index = X.index.copy()
        self.feature_columns = list(X.columns)
        return self

    def predict_proba(self, X):
        assert list(X.columns) == self.feature_columns
        out = np.zeros((len(X), 3), dtype=float)
        out[:, 0] = 0.65
        out[:, 1] = 0.20
        out[:, 2] = 0.15
        return out


def _m15(n=900):
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    close = 1.10 + np.sin(np.arange(n) / 10) * 0.001 + np.arange(n) * 0.00001
    return pd.DataFrame(
        {
            "open": close - 0.00005,
            "high": close + 0.00025,
            "low": close - 0.00025,
            "close": close,
            "tick_volume": np.arange(n) + 100,
        },
        index=idx,
    )


def test_generate_candle_oos_predictions_are_fold_local_and_auditable():
    models = []
    progress = []

    def model_factory(model_type):
        model = RecordingCandleModel()
        models.append(model)
        return model

    cfg = V5CandleOOSConfig(
        symbol="EURUSD",
        model_type="recording",
        train_days=5,
        test_days=2,
        max_folds=2,
        encoder_enabled=False,
    )

    result = generate_candle_oos_predictions(_m15(), cfg, model_factory=model_factory)

    assert len(result.folds) == 2
    assert set(result.predictions.columns).issuperset(
        {
            "fold",
            "prediction_time",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "candle_p_buy",
            "candle_p_hold",
            "candle_p_sell",
        }
    )
    assert_candle_predictions_are_oos(result.predictions)
    for fold, model in zip(result.folds, models):
        assert model.train_index.min() >= fold.train_start
        assert model.train_index.max() < fold.train_end

    generate_candle_oos_predictions(
        _m15(),
        cfg,
        model_factory=model_factory,
        progress_callback=lambda event, payload: progress.append((event, payload["fold"])),
    )

    assert ("fold_start", 0) in progress
    assert ("fold_done", 0) in progress


def test_candle_extra_features_do_not_use_incomplete_higher_timeframe_future():
    raw = _m15(96)
    prediction_time = pd.Timestamp("2026-01-01 01:15:00")
    X = pd.DataFrame({"base": [1.0]}, index=[prediction_time])

    baseline = add_candle_extra_features(raw, X)

    mutated = raw.copy()
    future_mask = (mutated.index > prediction_time) & (
        mutated.index < pd.Timestamp("2026-01-01 04:00:00")
    )
    mutated.loc[future_mask, "close"] = mutated.loc[future_mask, "close"] * 1.25
    changed = add_candle_extra_features(mutated, X)

    higher_tf_cols = ["ema_1h_ratio", "ema_1h_slope", "ema_4h_ratio", "ema_4h_slope"]
    pd.testing.assert_series_equal(
        baseline.loc[prediction_time, higher_tf_cols],
        changed.loc[prediction_time, higher_tf_cols],
        check_names=False,
    )
