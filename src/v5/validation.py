"""Validation guardrails for the V5 profitability track.

The functions here are small, deterministic checks that can be used by tests,
research scripts, and future report generators before a backtest result is
treated as deployable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Callable, Iterable

import pandas as pd
from pandas.testing import assert_frame_equal


class V5ValidationError(AssertionError):
    """Raised when a V5 profitability gate fails."""


def _as_feature_frame(result) -> pd.DataFrame:
    if isinstance(result, tuple):
        result = result[0]
    if not isinstance(result, pd.DataFrame):
        raise TypeError("feature builder must return a DataFrame or (DataFrame, labels)")
    return result.sort_index()


def assert_feature_builder_is_past_only(
    df: pd.DataFrame,
    build_features: Callable[[pd.DataFrame], pd.DataFrame | tuple[pd.DataFrame, object]],
    *,
    cutoff,
    perturbation_factor: float = 100.0,
) -> dict:
    """Verify rows at or before ``cutoff`` do not change when future rows change.

    This catches feature builders that accidentally use shift(-1), centered
    windows, full-sample statistics, or other future-dependent transformations.
    """

    baseline = _as_feature_frame(build_features(df.copy()))
    mutated = df.copy()
    future_mask = mutated.index > pd.Timestamp(cutoff)
    numeric_cols = mutated.select_dtypes(include="number").columns
    mutated.loc[future_mask, numeric_cols] = (
        mutated.loc[future_mask, numeric_cols] * perturbation_factor
    )
    changed = _as_feature_frame(build_features(mutated))

    common_index = baseline.index.intersection(changed.index)
    checked_index = common_index[common_index <= pd.Timestamp(cutoff)]
    common_cols = baseline.columns.intersection(changed.columns)
    if len(checked_index) == 0:
        raise V5ValidationError("no feature rows exist at or before cutoff")
    if len(common_cols) == 0:
        raise V5ValidationError("no common feature columns to compare")

    left = baseline.loc[checked_index, common_cols]
    right = changed.loc[checked_index, common_cols]
    try:
        assert_frame_equal(left, right, check_dtype=False, check_exact=False, atol=1e-10, rtol=1e-10)
    except AssertionError as exc:
        raise V5ValidationError(
            "feature rows changed before the cutoff when only future rows changed"
        ) from exc

    return {"checked_rows": len(checked_index), "checked_columns": len(common_cols)}


def assert_labels_match_forward_returns(
    close: pd.Series,
    labels: pd.Series,
    *,
    horizon: int,
    threshold: float,
) -> dict:
    """Verify labels are exactly derived from close[t+horizon] / close[t] - 1."""

    future_return = close.shift(-horizon) / close - 1
    expected = pd.Series(0, index=close.index, dtype="int64")
    expected[future_return > threshold] = 1
    expected[future_return < -threshold] = -1
    expected = expected.reindex(labels.index).dropna().astype("int64")
    observed = labels.reindex(expected.index).dropna().astype("int64")
    expected = expected.reindex(observed.index)
    mismatches = observed[observed != expected]
    if not mismatches.empty:
        sample = ", ".join(str(x) for x in mismatches.index[:5])
        raise V5ValidationError(
            f"{len(mismatches)} label mismatches against forward returns; sample={sample}"
        )
    return {"checked_labels": len(observed)}


def assert_fold_fit_records_are_train_only(records: Iterable[dict]) -> dict:
    """Verify scaler/encoder/model fit windows do not extend beyond fold train windows."""

    checked = 0
    for record in records:
        checked += 1
        fold = record.get("fold", "?")
        component = record.get("component", "?")
        train_start = pd.Timestamp(record["train_start"])
        train_end = pd.Timestamp(record["train_end"])
        fit_start = pd.Timestamp(record["fit_start"])
        fit_end = pd.Timestamp(record["fit_end"])
        if fit_start < train_start:
            raise V5ValidationError(
                f"fold {fold} {component} fit_start {fit_start} is before train_start {train_start}"
            )
        if fit_end > train_end:
            raise V5ValidationError(
                f"fold {fold} {component} fit_end {fit_end} is after train_end {train_end}"
            )
    return {"checked_fit_records": checked}


def assert_candle_predictions_are_oos(predictions: pd.DataFrame) -> dict:
    """Verify candle feature rows are produced only inside each fold's OOS test window."""

    required = {"fold", "prediction_time", "train_start", "train_end", "test_start", "test_end"}
    missing = required.difference(predictions.columns)
    if missing:
        raise V5ValidationError(f"missing candle OOS columns: {sorted(missing)}")

    frame = predictions.copy()
    for col in ["prediction_time", "train_start", "train_end", "test_start", "test_end"]:
        frame[col] = pd.to_datetime(frame[col])

    duplicated = frame.duplicated(subset=["prediction_time"], keep=False)
    if duplicated.any():
        raise V5ValidationError(
            f"{int(duplicated.sum())} duplicate candle prediction timestamps found"
        )

    outside = (frame["prediction_time"] < frame["test_start"]) | (
        frame["prediction_time"] >= frame["test_end"]
    )
    if outside.any():
        raise V5ValidationError(
            f"{int(outside.sum())} candle predictions are outside OOS test window"
        )

    in_train = (frame["prediction_time"] >= frame["train_start"]) & (
        frame["prediction_time"] < frame["train_end"]
    )
    if in_train.any():
        raise V5ValidationError(
            f"{int(in_train.sum())} candle predictions overlap the train window"
        )

    return {"checked_predictions": len(frame)}


@dataclass(frozen=True)
class BrokerExecutionRules:
    """Broker-realistic knobs shared by V5 replay and reconciliation scripts."""

    pip_size: float
    spread_pips: float
    commission_pips: float
    slippage_pips: float
    entry_delay_bars: int
    min_lot: float
    lot_step: float
    max_lot: float

    def __post_init__(self) -> None:
        if self.pip_size <= 0:
            raise ValueError("pip_size must be positive")
        if self.lot_step <= 0:
            raise ValueError("lot_step must be positive")
        if self.min_lot < 0 or self.max_lot < self.min_lot:
            raise ValueError("lot bounds are invalid")
        if self.entry_delay_bars < 0:
            raise ValueError("entry_delay_bars cannot be negative")

    @property
    def round_trip_cost_pips(self) -> float:
        return self.spread_pips + self.commission_pips + self.slippage_pips

    def normalize_lot(self, requested_lot: float) -> float:
        if requested_lot < self.min_lot:
            return 0.0
        capped = min(requested_lot, self.max_lot)
        steps = floor((capped + 1e-12) / self.lot_step)
        normalized = steps * self.lot_step
        if normalized < self.min_lot:
            return 0.0
        decimals = max(0, len(f"{self.lot_step:.10f}".rstrip("0").split(".")[-1]))
        return round(normalized, decimals)

