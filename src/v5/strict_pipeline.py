"""Strict fold-local walk-forward runner for V5 validation.

This module intentionally does not replace ``PredictorPipeline.walk_forward``.
It is the auditable V5 path where every learned component is fitted inside the
fold train window before producing out-of-sample test signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.features.candle_tokenizer import CandleTokenizer
from src.features.feature_pipeline import FeaturePipeline
from src.features.latent_encoder import LatentEncoder
from src.models.model_registry import _build_model
from src.pipeline import PipelineConfig
from src.v5.folds import FoldWindow, component_fit_records, expanding_fold_windows
from src.v5.validation import assert_fold_fit_records_are_train_only


@dataclass
class StrictFoldResult:
    window: FoldWindow
    n_train_rows: int
    n_test_rows: int
    test_index: pd.Index


@dataclass
class StrictWalkForwardResult:
    signals: pd.DataFrame
    folds: list[StrictFoldResult]
    fit_records: list[dict]


def run_strict_walk_forward(
    df_raw: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    model_factory: Callable[[str], object] | None = None,
    oos_candle_features: pd.DataFrame | None = None,
    max_folds: int | None = None,
) -> StrictWalkForwardResult:
    """Run strict fold-local feature fitting, model training, and signal output."""

    factory = model_factory or _build_model
    windows = expanding_fold_windows(
        df_raw.index,
        train_days=cfg.wf_train_days,
        test_days=cfg.wf_test_days,
    )
    signal_frames: list[pd.DataFrame] = []
    fold_results: list[StrictFoldResult] = []
    fit_records: list[dict] = []

    for window in windows:
        if max_folds is not None and len(fold_results) >= max_folds:
            break
        train_raw, test_raw = window.slice(df_raw)
        if len(train_raw) == 0 or len(test_raw) == 0:
            continue

        X_train, y_train, X_test = _build_fold_features(
            train_raw,
            test_raw,
            cfg,
            oos_candle_features=oos_candle_features,
        )
        if len(X_train) == 0 or len(X_test) == 0:
            continue

        y_train = y_train.reindex(X_train.index).dropna()
        X_train = X_train.reindex(y_train.index)
        if len(X_train) == 0:
            continue

        model = factory(cfg.model_type)
        model.train(X_train, y_train)
        fold_signals = _predict_signals(model, X_test, threshold=cfg.bt_threshold)
        signal_frames.append(fold_signals)
        fold_results.append(
            StrictFoldResult(
                window=window,
                n_train_rows=len(X_train),
                n_test_rows=len(X_test),
                test_index=X_test.index,
            )
        )
        components = ["feature_scaler"]
        if cfg.encoder_enabled:
            components.append("encoder")
        if cfg.candle_tokenizer_enabled:
            components.append("candle_tokenizer")
        if oos_candle_features is not None:
            components.append("candle_features")
        components.append("classifier")
        fit_records.extend(component_fit_records(window, components))

    assert_fold_fit_records_are_train_only(fit_records)
    signals = (
        pd.concat(signal_frames).sort_index()
        if signal_frames
        else pd.DataFrame(columns=["P_buy", "P_hold", "P_sell", "confidence", "signal"])
    )
    return StrictWalkForwardResult(
        signals=signals,
        folds=fold_results,
        fit_records=fit_records,
    )


def _build_fold_features(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    cfg: PipelineConfig,
    *,
    oos_candle_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    fp = FeaturePipeline(
        label_horizon=cfg.label_horizon,
        label_threshold=cfg.label_threshold,
        scale=cfg.scale,
        fractal_enabled=cfg.fractal_enabled,
        fractal_min_win=cfg.fractal_min_win,
        fractal_max_win=cfg.fractal_max_win,
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
            early_stopping_patience=cfg.encoder_patience,
            multitask_alpha=cfg.encoder_multitask_alpha,
            transformer_d_model=cfg.encoder_d_model,
            transformer_n_heads=cfg.encoder_n_heads,
            transformer_n_layers=cfg.encoder_n_layers,
            forecast_horizons=cfg.encoder_forecast_horizons,
            contrastive_temp=cfg.encoder_contrastive_temp,
            contrastive_proj_dim=cfg.encoder_proj_dim,
        )
        needs_labels = enc.mode in ("supervised", "transformer", "multitask")
        enc.fit(train_raw, y=y_train if needs_labels else None)
        latent_fold = enc.transform(fold_raw)
        X_train = _join_latent(X_train, latent_fold)
        X_test = _join_latent(X_test, latent_fold)

    if cfg.candle_tokenizer_enabled:
        tokenizer = CandleTokenizer(n_clusters=cfg.candle_tokenizer_clusters)
        tokenizer.fit(train_raw)
        clusters = tokenizer.transform(fold_raw)
        X_train = _join_extra(X_train, clusters)
        X_test = _join_extra(X_test, clusters)

    if oos_candle_features is not None:
        candle = _normalize_candle_features(oos_candle_features)
        X_train = _join_extra(X_train, candle)
        X_test = _join_extra(X_test, candle)

    y_train = y_train.reindex(X_train.index)
    X_test = X_test.reindex(columns=X_train.columns)
    return X_train, y_train, X_test


def _join_latent(X: pd.DataFrame, latent: pd.DataFrame) -> pd.DataFrame:
    shared = X.index.intersection(latent.index)
    return pd.concat([X.loc[shared], latent.loc[shared]], axis=1)


def _join_extra(X: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    shared = X.index.intersection(extra.index)
    return pd.concat([X.loc[shared], extra.loc[shared]], axis=1)


def _normalize_candle_features(candle: pd.DataFrame) -> pd.DataFrame:
    required = ["candle_p_buy", "candle_p_sell"]
    missing = [col for col in required if col not in candle.columns]
    if missing:
        raise ValueError(f"missing OOS candle feature columns: {missing}")
    out = candle[required].copy()
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _predict_signals(model, X_test: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    proba = model.predict_proba(X_test)
    if proba.ndim == 1:
        proba = proba.reshape(1, -1)
    out = pd.DataFrame(proba, index=X_test.index, columns=["P_buy", "P_hold", "P_sell"])
    out["confidence"] = out[["P_buy", "P_sell"]].max(axis=1)
    out["signal"] = "hold"
    out.loc[
        (out["P_buy"] >= threshold) & (out["P_buy"] > out["P_sell"]),
        "signal",
    ] = "buy"
    out.loc[
        (out["P_sell"] >= threshold) & (out["P_sell"] > out["P_buy"]),
        "signal",
    ] = "sell"
    return out
