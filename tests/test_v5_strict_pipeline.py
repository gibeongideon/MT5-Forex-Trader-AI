import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import PipelineConfig
from src.v5.strict_pipeline import run_strict_walk_forward
from src.v5.validation import assert_fold_fit_records_are_train_only


class RecordingModel:
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
        out[:, 1] = 0.70
        out[:, 0] = 0.20
        out[:, 2] = 0.10
        return out


def _ohlcv(n=140):
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    close = 1.10 + np.sin(np.arange(n) / 8) * 0.003 + np.arange(n) * 0.0001
    return pd.DataFrame(
        {
            "open": close - 0.0001,
            "high": close + 0.0005,
            "low": close - 0.0005,
            "close": close,
            "tick_volume": np.arange(n) + 1000,
        },
        index=idx,
    )


def test_strict_walk_forward_fits_model_only_on_each_train_window():
    models = []

    def model_factory(model_type):
        model = RecordingModel()
        models.append(model)
        return model

    cfg = PipelineConfig(
        model_type="recording",
        encoder_enabled=False,
        scale=True,
        label_horizon=2,
        label_threshold=0.0001,
        wf_train_days=70,
        wf_test_days=20,
        bt_threshold=0.6,
    )

    result = run_strict_walk_forward(_ohlcv(), cfg, model_factory=model_factory)

    assert len(result.folds) >= 1
    assert len(models) == len(result.folds)
    assert_fold_fit_records_are_train_only(result.fit_records)
    for fold_result, model in zip(result.folds, models):
        assert model.train_index.min() >= fold_result.window.train_start
        assert model.train_index.max() < fold_result.window.train_end
        assert result.signals.loc[fold_result.test_index, "signal"].eq("hold").all()


def test_strict_walk_forward_records_encoder_only_when_enabled():
    cfg = PipelineConfig(
        model_type="recording",
        encoder_enabled=False,
        scale=True,
        label_horizon=2,
        label_threshold=0.0001,
        wf_train_days=70,
        wf_test_days=20,
        bt_threshold=0.6,
    )

    result = run_strict_walk_forward(_ohlcv(), cfg, model_factory=lambda _: RecordingModel())
    components = {record["component"] for record in result.fit_records}

    assert "feature_scaler" in components
    assert "classifier" in components
    assert "encoder" not in components
