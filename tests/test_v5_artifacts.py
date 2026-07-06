import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.v5.artifacts import V5ArtifactWriter


def test_artifact_writer_creates_lumibot_style_run_bundle(tmp_path):
    writer = V5ArtifactWriter(tmp_path / "runs")
    run_dir = writer.write_run(
        run_id="EURUSD-candle-trail-smoke",
        settings={"symbol": "EURUSD", "mode": "candle_trail"},
        trades=[
            {
                "entry_time": "2026-01-01T00:00:00",
                "exit_time": "2026-01-01T00:15:00",
                "symbol": "EURUSD",
                "direction": "buy",
                "pnl_pips": 12.5,
            }
        ],
        equity=pd.Series(
            [10000.0, 10012.5],
            index=pd.to_datetime(["2026-01-01T00:00:00", "2026-01-01T00:15:00"]),
            name="equity",
        ),
        stats={"sharpe": 1.25, "max_drawdown": 0.02},
        folds=[{"fold": 0, "train_end": "2026-01-01", "test_end": "2026-02-01"}],
        reconciliation={"status": "dry_run_only", "matched_orders": 0},
    )

    assert run_dir.name == "EURUSD-candle-trail-smoke"
    assert json.loads((run_dir / "settings.json").read_text())["symbol"] == "EURUSD"
    assert json.loads((run_dir / "stats.json").read_text())["sharpe"] == 1.25
    assert json.loads((run_dir / "reconciliation.json").read_text())["status"] == "dry_run_only"
    assert pd.read_csv(run_dir / "trades.csv").iloc[0]["direction"] == "buy"
    assert pd.read_csv(run_dir / "equity.csv").iloc[-1]["equity"] == 10012.5
    assert pd.read_csv(run_dir / "folds.csv").iloc[0]["fold"] == 0
