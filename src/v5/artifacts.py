"""Lumibot-style run artifact writer for V5 research and paper gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


class V5ArtifactWriter:
    """Write one inspectable run directory per V5 validation or paper run."""

    def __init__(self, root: str | Path = "data/v5_runs"):
        self.root = Path(root)

    def write_run(
        self,
        *,
        run_id: str,
        settings: dict,
        trades: Iterable[dict],
        equity: pd.Series,
        stats: dict,
        folds: Iterable[dict] | None = None,
        reconciliation: dict | None = None,
    ) -> Path:
        run_dir = self.root / self._safe_run_id(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        self._write_json(run_dir / "settings.json", settings)
        self._write_json(run_dir / "stats.json", stats)
        self._write_json(run_dir / "reconciliation.json", reconciliation or {})

        pd.DataFrame(list(trades)).to_csv(run_dir / "trades.csv", index=False)
        equity.rename("equity").to_frame().reset_index(names="timestamp").to_csv(
            run_dir / "equity.csv", index=False
        )
        pd.DataFrame(list(folds or [])).to_csv(run_dir / "folds.csv", index=False)
        return run_dir

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")

    @staticmethod
    def _safe_run_id(run_id: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in run_id.strip())
        return safe or "run"
