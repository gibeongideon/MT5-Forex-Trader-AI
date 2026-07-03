import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.v5.leakage_report import build_leakage_proof, write_leakage_proof


def _signal_file(path: Path) -> None:
    idx = pd.date_range("2026-01-03", periods=4, freq="15min")
    pd.DataFrame(
        {
            "time": idx,
            "fold": [0, 0, 0, 0],
            "prediction_time": idx,
            "train_start": pd.Timestamp("2026-01-01"),
            "train_end": pd.Timestamp("2026-01-03"),
            "test_start": pd.Timestamp("2026-01-03"),
            "test_end": pd.Timestamp("2026-01-04"),
            "candle_p_buy": [0.8, 0.2, 0.1, 0.7],
            "candle_p_hold": [0.1, 0.7, 0.8, 0.2],
            "candle_p_sell": [0.1, 0.1, 0.1, 0.1],
        }
    ).to_parquet(path, index=False)


def _run_dir(path: Path, signal_path: Path) -> None:
    path.mkdir()
    equity = pd.Series(
        [10000, 10010, 10005, 10020],
        index=pd.date_range("2026-01-03", periods=4, freq="15min"),
        name="equity",
    )
    returns = equity.pct_change(fill_method=None).dropna()
    days = (equity.index[-1] - equity.index[0]).total_seconds() / 86_400
    sharpe = float(returns.mean() / returns.std() * math.sqrt(len(equity) / (days / 365.25)))
    (path / "settings.json").write_text(
        json.dumps({"symbol": "EURUSD", "signals_path": str(signal_path), "threshold": 0.6})
    )
    (path / "stats.json").write_text(
        json.dumps(
            {
                "symbol": "EURUSD",
                "sharpe": sharpe,
                "daily_sharpe": 1.1,
                "total_return": 0.01,
                "trades": 2,
                "max_drawdown": 0.001,
            }
        )
    )
    equity.to_frame().reset_index(names="timestamp").to_csv(path / "equity.csv", index=False)
    pd.DataFrame(
        {
            "symbol": ["EURUSD", "EURUSD"],
            "magic": [20260102, 20260102],
            "entry_time": ["2026-01-03 00:15:00", "2026-01-03 00:45:00"],
            "direction": ["buy", "buy"],
        }
    ).to_csv(path / "trades.csv", index=False)


def test_build_leakage_proof_validates_oos_and_recomputed_sharpe(tmp_path):
    signals = tmp_path / "signals.parquet"
    run = tmp_path / "run"
    _signal_file(signals)
    _run_dir(run, signals)

    proof = build_leakage_proof(run)

    assert proof["status"] == "leakage_checks_pass"
    assert proof["oos_predictions"]["status"] == "pass"
    assert proof["oos_predictions"]["folds"] == 1
    assert proof["oos_predictions"]["rows"] == 4
    assert proof["sharpe_check"]["status"] == "pass"
    assert proof["signal_file"]["sha256"]


def test_write_leakage_proof_creates_json_artifact(tmp_path):
    signals = tmp_path / "signals.parquet"
    run = tmp_path / "run"
    _signal_file(signals)
    _run_dir(run, signals)

    out = write_leakage_proof(run)

    payload = json.loads(out.read_text())
    assert out == run / "leakage_proof.json"
    assert payload["status"] == "leakage_checks_pass"
