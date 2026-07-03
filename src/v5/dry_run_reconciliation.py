"""Dry-run journal reconciliation for V5 paper-trading gates."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd


class DryRunReconciliationError(AssertionError):
    """Raised when dry-run journal rows do not match expected replay intents."""


def reconcile_dry_run_journal(
    *,
    expected_run_dir: str | Path,
    journal: pd.DataFrame | str | Path,
    tolerance_seconds: int = 60,
    confidence_tolerance: float = 0.05,
    lot_tolerance: float = 1e-9,
    fail_on_mismatch: bool = True,
) -> dict:
    """Compare expected replay trades against dry-run journal order intents."""

    run_dir = Path(expected_run_dir)
    expected = _load_expected(run_dir)
    observed = _load_journal(journal)
    observed = _filter_observed(observed, expected)

    matched_observed: set[int] = set()
    matches: list[dict] = []
    unmatched: list[dict] = []

    for exp_idx, exp in expected.iterrows():
        candidate_idx = _best_match(
            exp,
            observed.drop(index=list(matched_observed), errors="ignore"),
            tolerance_seconds=tolerance_seconds,
            confidence_tolerance=confidence_tolerance,
            lot_tolerance=lot_tolerance,
        )
        if candidate_idx is None:
            unmatched.append({"expected_index": int(exp_idx), "entry_time": str(exp["entry_time"])})
            continue
        matched_observed.add(candidate_idx)
        matches.append({"expected_index": int(exp_idx), "journal_index": int(candidate_idx)})

    unexpected = observed.drop(index=list(matched_observed), errors="ignore")
    report = {
        "status": "dry_run_reconciled",
        "expected_orders": int(len(expected)),
        "journal_orders": int(len(observed)),
        "matched_orders": int(len(matches)),
        "unmatched_expected": int(len(unmatched)),
        "unexpected_journal": int(len(unexpected)),
        "tolerance_seconds": tolerance_seconds,
        "confidence_tolerance": confidence_tolerance,
        "lot_tolerance": lot_tolerance,
        "matches": matches,
        "unmatched": unmatched[:20],
    }
    if unmatched or len(unexpected) > 0:
        report["status"] = "dry_run_mismatch"
        if fail_on_mismatch:
            raise DryRunReconciliationError(
                f"dry-run journal mismatch: unmatched={len(unmatched)} unexpected={len(unexpected)}"
            )
    return report


def write_reconciliation_report(report: dict, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    return path


def _load_expected(run_dir: Path) -> pd.DataFrame:
    trades = pd.read_csv(run_dir / "trades.csv")
    settings = json.loads((run_dir / "settings.json").read_text())
    if "symbol" not in trades.columns:
        trades["symbol"] = settings.get("broker_symbol") or settings.get("symbol")
    if "magic" not in trades.columns:
        trades["magic"] = settings.get("magic_number")
    for column in ["entry_time", "signal_time"]:
        if column in trades.columns:
            trades[column] = pd.to_datetime(trades[column])
    return trades


def _load_journal(journal: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(journal, pd.DataFrame):
        frame = journal.copy()
    else:
        path = Path(journal)
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        else:
            with sqlite3.connect(path) as conn:
                frame = pd.read_sql_query("SELECT * FROM trades ORDER BY id", conn)
    if "entry_time" in frame.columns:
        frame["entry_time"] = pd.to_datetime(frame["entry_time"])
    return frame


def _filter_observed(observed: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    frame = observed.copy()
    if "exit_reason" in frame.columns:
        dry_mask = frame["exit_reason"].astype(str).str.contains("dry|pending|open", case=False, na=False)
        frame = frame[dry_mask]
    symbols = set(expected["symbol"].dropna().astype(str))
    if "symbol" in frame.columns and symbols:
        frame = frame[frame["symbol"].astype(str).isin(symbols)]
    if "magic" in frame.columns and expected["magic"].notna().any():
        magics = set(expected["magic"].dropna().astype("int64"))
        frame = frame[frame["magic"].fillna(-1).astype("int64").isin(magics)]
    return frame


def _best_match(
    expected: pd.Series,
    observed: pd.DataFrame,
    *,
    tolerance_seconds: int,
    confidence_tolerance: float,
    lot_tolerance: float,
) -> int | None:
    for idx, row in observed.iterrows():
        if str(row.get("symbol")) != str(expected.get("symbol")):
            continue
        if str(row.get("direction")).lower() != str(expected.get("direction")).lower():
            continue
        if pd.notna(expected.get("magic")) and "magic" in row and int(row.get("magic")) != int(expected.get("magic")):
            continue
        if abs(float(row.get("volume", 0.0)) - float(expected.get("volume", 0.0))) > lot_tolerance:
            continue
        if abs(float(row.get("sl_pips", 0.0)) - float(expected.get("sl_pips", 0.0))) > 1e-9:
            continue
        if abs(float(row.get("tp_pips", 0.0)) - float(expected.get("tp_pips", 0.0))) > 1e-9:
            continue
        if abs(float(row.get("confidence", 0.0)) - float(expected.get("confidence", 0.0))) > confidence_tolerance:
            continue
        delta = abs((pd.Timestamp(row["entry_time"]) - pd.Timestamp(expected["entry_time"])).total_seconds())
        if delta > tolerance_seconds:
            continue
        return int(idx)
    return None
