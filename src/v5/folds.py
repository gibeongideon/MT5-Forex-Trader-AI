"""Fold-window utilities for strict V5 walk-forward validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class FoldWindow:
    """Half-open train/test window for one walk-forward fold."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def slice(self, frame: pd.DataFrame | pd.Series):
        train = frame[(frame.index >= self.train_start) & (frame.index < self.train_end)]
        test = frame[(frame.index >= self.test_start) & (frame.index < self.test_end)]
        return train, test

    def as_dict(self) -> dict:
        return {
            "fold": self.fold,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
        }


def expanding_fold_windows(
    index: pd.Index,
    *,
    train_days: int,
    test_days: int,
) -> list[FoldWindow]:
    """Create expanding half-open train/test windows from a DatetimeIndex."""

    if len(index) == 0:
        return []
    dates = pd.DatetimeIndex(index).sort_values()
    start = dates[0]
    train_end = start + pd.Timedelta(days=train_days)
    last = dates[-1]
    folds: list[FoldWindow] = []
    fold = 0
    while train_end < last:
        test_end = min(train_end + pd.Timedelta(days=test_days), last)
        if test_end - train_end < pd.Timedelta(days=test_days):
            break
        test_count = ((dates >= train_end) & (dates < test_end)).sum()
        train_count = (dates < train_end).sum()
        if train_count > 0 and test_count > 0:
            folds.append(
                FoldWindow(
                    fold=fold,
                    train_start=start,
                    train_end=train_end,
                    test_start=train_end,
                    test_end=test_end,
                )
            )
            fold += 1
        train_end = test_end
    return folds


def component_fit_records(fold: FoldWindow, components: Iterable[str]) -> list[dict]:
    """Record that each component was fit only over this fold's train window."""

    return [
        {
            "fold": fold.fold,
            "component": component,
            "train_start": fold.train_start,
            "train_end": fold.train_end,
            "fit_start": fold.train_start,
            "fit_end": fold.train_end,
        }
        for component in components
    ]
