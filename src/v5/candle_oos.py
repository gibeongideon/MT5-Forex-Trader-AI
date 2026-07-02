"""Fold-local OOS candle probability generation for V5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.features.feature_pipeline import FeaturePipeline
from src.features.latent_encoder import LatentEncoder
from src.models.model_registry import _build_model
from src.v5.folds import FoldWindow
from src.v5.validation import assert_candle_predictions_are_oos


@dataclass
class V5CandleOOSConfig:
    symbol: str
    model_type: str = "catboost"
    train_days: int = 120
    test_days: int = 60
    label_horizon: int = 1
    label_threshold: float = 0.0005
    encoder_enabled: bool = True
    encoder_mode: str = "supervised"
    encoder_latent_dim: int = 8
    encoder_window: int = 50
    encoder_epochs: int = 30
    encoder_batch: int = 4096
    encoder_lr: float = 1e-3
    max_folds: int | None = None


@dataclass
class V5CandleOOSResult:
    predictions: pd.DataFrame
    folds: list[FoldWindow]


def generate_candle_oos_predictions(
    df_raw: pd.DataFrame,
    cfg: V5CandleOOSConfig,
    *,
    model_factory: Callable[[str], object] | None = None,
    progress_callback: Callable[[str, dict], None] | None = None,
) -> V5CandleOOSResult:
    """Train candle models inside each fold and emit OOS probabilities.

    This is the leakage-hardened replacement for using cached candle fold models
    with a final full-data encoder. Each learned component is fitted inside the
    fold's train window before predicting that fold's test window.
    """

    raw = df_raw.sort_index()
    factory = model_factory or _build_model
    frames: list[pd.DataFrame] = []
    used_folds: list[FoldWindow] = []

    for window in sliding_fold_windows(
        raw.index,
        train_days=cfg.train_days,
        test_days=cfg.test_days,
    ):
        if cfg.max_folds is not None and len(used_folds) >= cfg.max_folds:
            break
        train_raw, test_raw = window.slice(raw)
        if len(train_raw) == 0 or len(test_raw) == 0:
            continue
        if progress_callback is not None:
            progress_callback(
                "fold_start",
                {
                    "fold": window.fold,
                    "train_start": window.train_start,
                    "train_end": window.train_end,
                    "test_start": window.test_start,
                    "test_end": window.test_end,
                    "train_rows": len(train_raw),
                    "test_rows": len(test_raw),
                },
            )

        X_train, y_train, X_test = _build_candle_features(train_raw, test_raw, cfg)
        y_train = y_train.reindex(X_train.index).dropna()
        X_train = X_train.reindex(y_train.index)
        if len(X_train) == 0 or len(X_test) == 0:
            continue

        model = factory(cfg.model_type)
        model.train(X_train, y_train)
        proba = model.predict_proba(X_test)
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)

        fold_predictions = pd.DataFrame(
            {
                "fold": window.fold,
                "prediction_time": X_test.index,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "candle_p_buy": proba[:, 0],
                "candle_p_hold": proba[:, 1],
                "candle_p_sell": proba[:, 2],
            },
            index=X_test.index,
        )
        frames.append(fold_predictions)
        used_folds.append(window)
        if progress_callback is not None:
            progress_callback(
                "fold_done",
                {
                    "fold": window.fold,
                    "train_rows": len(X_train),
                    "test_rows": len(X_test),
                    "prediction_rows": len(fold_predictions),
                },
            )

    predictions = (
        pd.concat(frames).sort_index()
        if frames
        else pd.DataFrame(
            columns=[
                "fold",
                "prediction_time",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "candle_p_buy",
                "candle_p_hold",
                "candle_p_sell",
            ]
        )
    )
    if len(predictions):
        assert_candle_predictions_are_oos(predictions)
    return V5CandleOOSResult(predictions=predictions, folds=used_folds)


def sliding_fold_windows(index: pd.Index, *, train_days: int, test_days: int) -> list[FoldWindow]:
    if len(index) == 0:
        return []
    dates = pd.DatetimeIndex(index).sort_values()
    train_delta = pd.Timedelta(days=train_days)
    test_delta = pd.Timedelta(days=test_days)
    train_end = dates[0] + train_delta
    last = dates[-1]
    folds: list[FoldWindow] = []
    fold = 0
    while train_end + test_delta <= last:
        train_start = train_end - train_delta
        test_end = train_end + test_delta
        train_count = ((dates >= train_start) & (dates < train_end)).sum()
        test_count = ((dates >= train_end) & (dates < test_end)).sum()
        if train_count > 0 and test_count > 0:
            folds.append(
                FoldWindow(
                    fold=fold,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=train_end,
                    test_end=test_end,
                )
            )
            fold += 1
        train_end = test_end
    return folds


def _build_candle_features(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    cfg: V5CandleOOSConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    fp = FeaturePipeline(
        label_horizon=cfg.label_horizon,
        label_threshold=cfg.label_threshold,
        scale=True,
    )
    X_train, y_train = fp.build(train_raw, fit=True)
    fold_raw = pd.concat([train_raw, test_raw])
    X_fold, _ = fp.build(fold_raw, fit=False)
    X_test = X_fold[X_fold.index >= test_raw.index[0]]

    if cfg.encoder_enabled:
        enc = LatentEncoder(
            mode=cfg.encoder_mode,
            latent_dim=cfg.encoder_latent_dim,
            window_size=cfg.encoder_window,
            epochs=cfg.encoder_epochs,
            batch_size=cfg.encoder_batch,
            lr=cfg.encoder_lr,
        )
        needs_labels = cfg.encoder_mode in ("supervised", "transformer", "multitask")
        enc.fit(train_raw, y=y_train if needs_labels else None)
        latent = enc.transform(fold_raw)
        X_train = _join_extra(X_train, latent)
        X_test = _join_extra(X_test, latent)

    X_train = add_candle_extra_features(fold_raw, X_train)
    X_test = add_candle_extra_features(fold_raw, X_test)
    X_test = X_test.reindex(columns=X_train.columns)
    return X_train, y_train.reindex(X_train.index), X_test


def add_candle_extra_features(df_raw: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    idx = X.index
    hour = idx.hour
    extra = pd.DataFrame(index=idx)
    extra["session_sydney"] = ((hour >= 22) | (hour < 7)).astype(float)
    extra["session_tokyo"] = ((hour >= 0) & (hour < 9)).astype(float)
    extra["session_london"] = ((hour >= 8) & (hour < 17)).astype(float)
    extra["session_ny"] = ((hour >= 13) & (hour < 22)).astype(float)
    extra["session_tok_lon"] = ((hour >= 8) & (hour < 9)).astype(float)
    extra["session_lon_ny"] = ((hour >= 13) & (hour < 17)).astype(float)
    extra["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    extra["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    close_1h = df_raw["close"].resample("1h").last().ffill()
    ema_1h = close_1h.ewm(span=20, adjust=False).mean()
    ema_1h_m15 = ema_1h.reindex(df_raw.index, method="ffill")
    extra["ema_1h_ratio"] = (
        (df_raw["close"] - ema_1h_m15) / df_raw["close"]
    ).reindex(idx).fillna(0)
    extra["ema_1h_slope"] = (ema_1h_m15.diff(4) / df_raw["close"]).reindex(idx).fillna(0)

    close_4h = df_raw["close"].resample("4h").last().ffill()
    ema_4h = close_4h.ewm(span=50, adjust=False).mean()
    ema_4h_m15 = ema_4h.reindex(df_raw.index, method="ffill")
    extra["ema_4h_ratio"] = (
        (df_raw["close"] - ema_4h_m15) / df_raw["close"]
    ).reindex(idx).fillna(0)
    extra["ema_4h_slope"] = (ema_4h_m15.diff(16) / df_raw["close"]).reindex(idx).fillna(0)

    return pd.concat([X, extra.reindex(idx).fillna(0)], axis=1)


def _join_extra(X: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    shared = X.index.intersection(extra.index)
    return pd.concat([X.loc[shared], extra.loc[shared]], axis=1)
