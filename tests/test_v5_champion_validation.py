import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import PipelineConfig
from src.v5.champion_validation import V5ChampionValidationConfig, run_champion_validation
from src.v5.validation import BrokerExecutionRules


class AlternatingModel:
    def train(self, X, y):
        self.columns = list(X.columns)
        return self

    def predict_proba(self, X):
        assert list(X.columns) == self.columns
        out = np.zeros((len(X), 3), dtype=float)
        out[:, 1] = 0.20
        out[::2, 0] = 0.75
        out[::2, 2] = 0.05
        out[1::2, 0] = 0.05
        out[1::2, 2] = 0.75
        return out


def _csv_data(path, n=260):
    idx = pd.date_range("2026-01-01", periods=n, freq="15min")
    close = 1.10 + np.sin(np.arange(n) / 6) * 0.001 + np.arange(n) * 0.00002
    frame = pd.DataFrame(
        {
            "time": idx,
            "open": close - 0.00005,
            "high": close + 0.00035,
            "low": close - 0.00035,
            "close": close,
            "tick_volume": np.arange(n) + 100,
        }
    )
    frame.to_csv(path, index=False)


def test_champion_validation_writes_artifact_bundle(tmp_path):
    data_path = tmp_path / "EURUSD_M15.csv"
    _csv_data(data_path)
    cfg = PipelineConfig(
        model_type="alternating",
        encoder_enabled=False,
        scale=True,
        label_horizon=2,
        label_threshold=0.0001,
        wf_train_days=1,
        wf_test_days=1,
        bt_threshold=0.6,
    )
    rules = BrokerExecutionRules(
        pip_size=0.0001,
        spread_pips=1.0,
        commission_pips=0.5,
        slippage_pips=0.0,
        entry_delay_bars=1,
        min_lot=0.01,
        lot_step=0.01,
        max_lot=0.50,
    )
    run_cfg = V5ChampionValidationConfig(
        symbol="EURUSD",
        data_path=data_path,
        run_id="unit-eurusd",
        artifact_root=tmp_path / "v5_runs",
        requested_lot=0.026,
        sl_pips=10,
        tp_pips=20,
        pipeline=cfg,
        broker_rules=rules,
    )

    result = run_champion_validation(run_cfg, model_factory=lambda _: AlternatingModel())

    assert result.run_dir == tmp_path / "v5_runs" / "unit-eurusd"
    assert result.strict.folds
    assert result.replay.trades
    settings = json.loads((result.run_dir / "settings.json").read_text())
    stats = json.loads((result.run_dir / "stats.json").read_text())
    trades = pd.read_csv(result.run_dir / "trades.csv")
    folds = pd.read_csv(result.run_dir / "folds.csv")
    reconciliation = json.loads((result.run_dir / "reconciliation.json").read_text())

    assert settings["symbol"] == "EURUSD"
    assert settings["data_path"].endswith("EURUSD_M15.csv")
    assert stats["folds"] == len(result.strict.folds)
    assert stats["signals"] == len(result.strict.signals)
    assert len(trades) == len(result.replay.trades)
    assert set(folds["component"]).issuperset({"feature_scaler", "classifier"})
    assert reconciliation["status"] == "research_replay_only"


def test_champion_validation_injects_oos_candle_feature_file(tmp_path):
    data_path = tmp_path / "EURUSD_M15.csv"
    _csv_data(data_path)
    raw = pd.read_csv(data_path)
    candle = pd.DataFrame(
        {
            "time": pd.to_datetime(raw["time"]),
            "candle_p_buy": np.linspace(0.2, 0.8, len(raw)),
            "candle_p_sell": np.linspace(0.8, 0.2, len(raw)),
        }
    )
    candle_path = tmp_path / "candle_signal_EURUSD.parquet"
    candle.to_parquet(candle_path, index=False)
    cfg = PipelineConfig(
        model_type="alternating",
        encoder_enabled=False,
        scale=True,
        label_horizon=2,
        label_threshold=0.0001,
        wf_train_days=1,
        wf_test_days=1,
        bt_threshold=0.6,
    )
    rules = BrokerExecutionRules(
        pip_size=0.0001,
        spread_pips=1.0,
        commission_pips=0.5,
        slippage_pips=0.0,
        entry_delay_bars=1,
        min_lot=0.01,
        lot_step=0.01,
        max_lot=0.50,
    )
    run_cfg = V5ChampionValidationConfig(
        symbol="EURUSD",
        data_path=data_path,
        run_id="unit-hybrid",
        artifact_root=tmp_path / "v5_runs",
        requested_lot=0.01,
        sl_pips=10,
        tp_pips=20,
        pipeline=cfg,
        broker_rules=rules,
        candle_features_path=candle_path,
    )

    result = run_champion_validation(run_cfg, model_factory=lambda _: AlternatingModel())

    settings = json.loads((result.run_dir / "settings.json").read_text())
    folds = pd.read_csv(result.run_dir / "folds.csv")
    assert settings["candle_features_path"].endswith("candle_signal_EURUSD.parquet")
    assert "candle_features" in set(folds["component"])
