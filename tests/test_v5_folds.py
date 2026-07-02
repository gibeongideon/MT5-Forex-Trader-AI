import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.v5.folds import FoldWindow, component_fit_records, expanding_fold_windows
from src.v5.validation import assert_fold_fit_records_are_train_only


def test_expanding_fold_windows_match_train_and_test_boundaries():
    idx = pd.date_range("2026-01-01", periods=10, freq="D")

    folds = expanding_fold_windows(idx, train_days=4, test_days=2)

    assert folds == [
        FoldWindow(
            fold=0,
            train_start=idx[0],
            train_end=idx[4],
            test_start=idx[4],
            test_end=idx[6],
        ),
        FoldWindow(
            fold=1,
            train_start=idx[0],
            train_end=idx[6],
            test_start=idx[6],
            test_end=idx[8],
        ),
    ]


def test_fold_slices_are_non_overlapping_and_half_open():
    idx = pd.date_range("2026-01-01", periods=8, freq="D")
    frame = pd.DataFrame({"close": range(len(idx))}, index=idx)
    fold = FoldWindow(
        fold=0,
        train_start=idx[0],
        train_end=idx[4],
        test_start=idx[4],
        test_end=idx[7],
    )

    train, test = fold.slice(frame)

    assert train.index.tolist() == list(idx[:4])
    assert test.index.tolist() == list(idx[4:7])
    assert train.index.max() < test.index.min()


def test_component_fit_records_feed_existing_train_only_guard():
    fold = FoldWindow(
        fold=2,
        train_start=pd.Timestamp("2026-01-01"),
        train_end=pd.Timestamp("2026-02-01"),
        test_start=pd.Timestamp("2026-02-01"),
        test_end=pd.Timestamp("2026-03-01"),
    )

    records = component_fit_records(fold, ["feature_scaler", "encoder", "classifier"])

    assert records[0]["component"] == "feature_scaler"
    assert records[-1]["fit_end"] == fold.train_end
    assert_fold_fit_records_are_train_only(records)
