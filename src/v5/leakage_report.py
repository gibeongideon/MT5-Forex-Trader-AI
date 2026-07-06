"""Leakage and Sharpe evidence reports for V5 candidate runs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from src.v5.validation import assert_candle_predictions_are_oos


def build_leakage_proof(
    run_dir: str | Path,
    *,
    signal_path: str | Path | None = None,
    stress_run_dirs: list[str | Path] | None = None,
    sharpe_tolerance: float = 1e-6,
) -> dict:
    """Build an inspectable report showing why a V5 Sharpe is trusted as OOS."""

    run_path = Path(run_dir)
    settings = _read_json(run_path / "settings.json")
    stats = _read_json(run_path / "stats.json")
    resolved_signal = Path(signal_path or settings["signals_path"])
    signals = _read_signal_file(resolved_signal)
    assert_candle_predictions_are_oos(signals)

    equity = _read_equity(run_path / "equity.csv")
    recomputed_sharpe = _annualized_sharpe(equity)
    reported_sharpe = float(stats.get("sharpe", 0.0))
    sharpe_delta = abs(reported_sharpe - recomputed_sharpe)
    sharpe_status = "pass" if sharpe_delta <= sharpe_tolerance else "fail"

    proof = {
        "status": "leakage_checks_pass" if sharpe_status == "pass" else "leakage_checks_fail",
        "run_dir": str(run_path),
        "symbol": stats.get("symbol") or settings.get("symbol"),
        "run_stats": {
            "trades": int(stats.get("trades", 0)),
            "total_return": float(stats.get("total_return", 0.0)),
            "max_drawdown": float(stats.get("max_drawdown", 0.0)),
            "sharpe": reported_sharpe,
            "daily_sharpe": float(stats.get("daily_sharpe", 0.0)),
        },
        "signal_file": {
            "path": str(resolved_signal),
            "sha256": _sha256(resolved_signal),
        },
        "oos_predictions": _oos_summary(signals),
        "sharpe_check": {
            "status": sharpe_status,
            "reported_sharpe": reported_sharpe,
            "recomputed_sharpe": recomputed_sharpe,
            "absolute_delta": sharpe_delta,
            "tolerance": sharpe_tolerance,
        },
        "stress_runs": [_stress_summary(Path(path)) for path in (stress_run_dirs or [])],
        "interpretation": (
            "Passed OOS metadata and Sharpe recomputation checks. This supports that the "
            "reported Sharpe comes from fold-local out-of-sample candle probabilities and "
            "the persisted equity curve, but it is still research-only until dry-run and "
            "demo-live reconciliation pass."
        ),
    }
    return proof


def write_leakage_proof(
    run_dir: str | Path,
    *,
    signal_path: str | Path | None = None,
    stress_run_dirs: list[str | Path] | None = None,
    out_path: str | Path | None = None,
) -> Path:
    run_path = Path(run_dir)
    proof = build_leakage_proof(
        run_path,
        signal_path=signal_path,
        stress_run_dirs=stress_run_dirs,
    )
    out = Path(out_path) if out_path else run_path / "leakage_proof.json"
    out.write_text(json.dumps(proof, indent=2, sort_keys=True, default=str) + "\n")
    return out


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _read_signal_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    if "prediction_time" in frame.columns:
        frame["prediction_time"] = pd.to_datetime(frame["prediction_time"])
    for column in ["train_start", "train_end", "test_start", "test_end"]:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column])
    return frame


def _read_equity(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    timestamp_col = "timestamp" if "timestamp" in frame.columns else frame.columns[0]
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col])
    return pd.Series(frame["equity"].astype(float).to_numpy(), index=frame[timestamp_col], name="equity")


def _oos_summary(signals: pd.DataFrame) -> dict:
    frame = signals.copy()
    frame["prediction_time"] = pd.to_datetime(frame["prediction_time"])
    by_fold = frame.groupby("fold").agg(
        rows=("prediction_time", "count"),
        first_prediction=("prediction_time", "min"),
        last_prediction=("prediction_time", "max"),
        train_start=("train_start", "min"),
        train_end=("train_end", "max"),
        test_start=("test_start", "min"),
        test_end=("test_end", "max"),
    )
    return {
        "status": "pass",
        "rows": int(len(frame)),
        "folds": int(frame["fold"].nunique()),
        "duplicate_prediction_times": int(frame.duplicated(subset=["prediction_time"]).sum()),
        "first_prediction": str(frame["prediction_time"].min()),
        "last_prediction": str(frame["prediction_time"].max()),
        "folds_table": by_fold.reset_index().to_dict(orient="records"),
    }


def _stress_summary(run_path: Path) -> dict:
    stats = _read_json(run_path / "stats.json")
    settings = _read_json(run_path / "settings.json")
    return {
        "run_id": run_path.name,
        "threshold": settings.get("threshold"),
        "entry_delay_bars": settings.get("broker_rules", {}).get("entry_delay_bars"),
        "spread_pips": settings.get("broker_rules", {}).get("spread_pips"),
        "slippage_pips": settings.get("broker_rules", {}).get("slippage_pips"),
        "trades": int(stats.get("trades", 0)),
        "total_return": float(stats.get("total_return", 0.0)),
        "sharpe": float(stats.get("sharpe", 0.0)),
        "max_drawdown": float(stats.get("max_drawdown", 0.0)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _annualized_sharpe(equity: pd.Series) -> float:
    returns = equity.pct_change(fill_method=None).dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86_400, 1e-9)
    bars_per_year = float(len(equity) / (days / 365.25))
    return float(returns.mean() / returns.std() * math.sqrt(bars_per_year))
