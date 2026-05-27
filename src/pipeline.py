"""
PredictorPipeline — end-to-end configurable prediction pipeline.

Orchestrates the full stack in one object:
    FeaturePipeline → LatentEncoder (optional) → Model → RiskManager

One class. One config section. One artifact directory.

Quick start:
    pipe = PredictorPipeline.from_config()       # reads config.yaml pipeline:
    X, y = pipe.build_features(df_raw)           # engineer features + encode
    result = pipe.walk_forward(X, y, prices)     # evaluate
    result.report()

    pipe.fit_full(X, y)                          # train on all data
    pipe.save()                                  # persist artifacts

    pipe2 = PredictorPipeline.from_config()
    pipe2.load()
    signal = pipe2.predict(ohlcv_df)             # live inference
    # {"signal": "buy", "confidence": 0.71, "P_buy": 0.71, ...}

See config.yaml → pipeline: for all knobs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.backtester import BacktestConfig
from src.features.feature_pipeline import FeaturePipeline
from src.features.latent_encoder import LatentEncoder
from src.model_registry import _build_model
from src.risk_manager import RiskManager, RiskConfig
from src.walk_forward import WalkForwardConfig, WalkForwardResult, WalkForwardValidator


# ── Pipeline config ────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """All pipeline knobs in one place. Every field has a sensible default."""

    # Model
    model_type:  str   = "xgboost"     # xgboost | lightgbm | catboost | ensemble

    # Data
    data_path:   str   = "data/EURUSD_M15.csv"
    train_frac:  float = 0.8            # fraction used to fit scaler + encoder

    # Feature engineering
    label_horizon:   int   = 4
    label_threshold: float = 0.0003
    scale:           bool  = True       # StandardScaler on base features

    # Latent encoder
    encoder_enabled:    bool  = True
    encoder_mode:       str   = "supervised"    # supervised | autoencoder
    encoder_latent_dim: int   = 8
    encoder_window:     int   = 50              # OHLCV bars per input window
    encoder_epochs:     int   = 30
    encoder_batch:      int   = 4096
    encoder_lr:         float = 1e-3

    # Walk-forward
    wf_window_type: str          = "expanding"  # expanding | sliding
    wf_train_days:  int          = 180
    wf_test_days:   int          = 30
    wf_cache_dir:   Optional[str] = "data/models/pipeline_wf_cache"

    # Backtester
    bt_threshold:  float = 0.40
    bt_sl_pips:    float = 30.0
    bt_tp_pips:    float = 60.0
    bt_spread:     float = 1.0
    bt_balance:    float = 10_000.0
    bt_risk_pct:   float = 0.01
    bt_use_regime: bool  = False

    # Artifact storage
    artifacts_dir: str = "data/models/pipeline"

    # ── Risk manager tiers (mirrors config.yaml risk_manager.tiers) ────────
    rm_tiers: list = None   # None → use RiskConfig defaults

    def __post_init__(self):
        if self.rm_tiers is None:
            self.rm_tiers = [
                [0.75, 0.020],
                [0.65, 0.015],
                [0.55, 0.0075],
                [0.40, 0.005],
            ]

    @classmethod
    def from_dict(cls, d: dict, rm_cfg: dict = None) -> "PipelineConfig":
        """Build from the pipeline: section of config.yaml."""
        enc  = d.get("encoder",       {})
        feat = d.get("features",      {})
        wf   = d.get("walk_forward",  {})
        bt   = d.get("backtest",      {})
        art  = d.get("artifacts",     {})
        rm   = rm_cfg or {}

        tiers = rm.get("tiers", None)

        return cls(
            model_type  = d.get("model_type",  "xgboost"),
            data_path   = d.get("data_path",   "data/EURUSD_M15.csv"),
            train_frac  = d.get("train_frac",  0.8),

            label_horizon   = feat.get("label_horizon",   4),
            label_threshold = feat.get("label_threshold", 0.0003),
            scale           = feat.get("scale",           True),

            encoder_enabled    = enc.get("enabled",     True),
            encoder_mode       = enc.get("mode",        "supervised"),
            encoder_latent_dim = enc.get("latent_dim",  8),
            encoder_window     = enc.get("window_size", 50),
            encoder_epochs     = enc.get("epochs",      30),
            encoder_batch      = enc.get("batch_size",  4096),
            encoder_lr         = enc.get("lr",          1e-3),

            wf_window_type = wf.get("window_type", "expanding"),
            wf_train_days  = wf.get("train_days",  180),
            wf_test_days   = wf.get("test_days",   30),
            wf_cache_dir   = wf.get("cache_dir",   "data/models/pipeline_wf_cache"),

            bt_threshold   = bt.get("threshold",        0.40),
            bt_sl_pips     = bt.get("sl_pips",          30.0),
            bt_tp_pips     = bt.get("tp_pips",          60.0),
            bt_spread      = bt.get("spread_pips",      1.0),
            bt_balance     = bt.get("initial_balance",  10_000.0),
            bt_risk_pct    = bt.get("risk_pct",         0.01),
            bt_use_regime  = bt.get("use_regime_filter", False),

            artifacts_dir = art.get("directory", "data/models/pipeline"),
            rm_tiers      = tiers,
        )

    def summary(self) -> str:
        enc_str = (
            f"{self.encoder_mode}  latent_dim={self.encoder_latent_dim}"
            f"  window={self.encoder_window}  epochs={self.encoder_epochs}"
            if self.encoder_enabled else "disabled"
        )
        return (
            f"PipelineConfig\n"
            f"  model        : {self.model_type}\n"
            f"  data         : {self.data_path}  (train_frac={self.train_frac})\n"
            f"  features     : horizon={self.label_horizon}  threshold={self.label_threshold}"
            f"  scale={self.scale}\n"
            f"  encoder      : {enc_str}\n"
            f"  walk_forward : {self.wf_window_type}  train={self.wf_train_days}d"
            f"  test={self.wf_test_days}d\n"
            f"  backtest     : threshold={self.bt_threshold}  SL={self.bt_sl_pips}p"
            f"  TP={self.bt_tp_pips}p  spread={self.bt_spread}p\n"
            f"  artifacts    : {self.artifacts_dir}"
        )


# ── Pipeline ───────────────────────────────────────────────────────────────────

class PredictorPipeline:
    """
    End-to-end configurable prediction pipeline.

    Lifecycle
    ---------
    1. build_features(df_raw)  — compute X, y; fits scaler + encoder on train portion
    2. walk_forward(X, y, prices) — walk-forward evaluation (returns WalkForwardResult)
    3. fit_full(X, y)          — train model on the complete feature matrix
    4. save()                  — write scaler + encoder + model + meta to artifacts_dir
    5. predict(df_raw)         — live inference; load() first if needed

    All behaviour is controlled by PipelineConfig (parsed from config.yaml).
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

        self._fp = FeaturePipeline(
            label_horizon   = cfg.label_horizon,
            label_threshold = cfg.label_threshold,
            scale           = cfg.scale,
        )

        self._enc: Optional[LatentEncoder] = (
            LatentEncoder(
                mode        = cfg.encoder_mode,
                latent_dim  = cfg.encoder_latent_dim,
                window_size = cfg.encoder_window,
                epochs      = cfg.encoder_epochs,
                batch_size  = cfg.encoder_batch,
                lr          = cfg.encoder_lr,
            )
            if cfg.encoder_enabled else None
        )

        self._model = None
        self._rm = self._build_rm()

        # State set during build_features / load
        self._feature_cols: list[str] = []
        self._split_date: Optional[pd.Timestamp] = None

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, path: str = "config.yaml") -> "PredictorPipeline":
        """
        Build pipeline from config.yaml.
        Reads pipeline: section for pipeline config and risk_manager: for tiers.
        """
        with open(path) as f:
            full = yaml.safe_load(f)
        cfg = PipelineConfig.from_dict(
            full.get("pipeline", {}),
            rm_cfg=full.get("risk_manager", {}),
        )
        return cls(cfg)

    # ── Step 1: Feature engineering ───────────────────────────────────────────

    def build_features(
        self,
        df_raw:     pd.DataFrame,
        train_frac: float = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build the full feature matrix (base indicators + optional latent dims).

        - StandardScaler fitted on [0 : train_frac] portion only.
        - Encoder trained on [0 : train_frac] portion only.
        - Returns X (full length, aligned) and y (labels).

        Parameters
        ----------
        df_raw     : raw OHLCV DataFrame
        train_frac : override cfg.train_frac if provided

        Returns
        -------
        X : pd.DataFrame  — feature matrix
        y : pd.Series     — labels (-1 / 0 / 1)
        """
        frac  = train_frac if train_frac is not None else self.cfg.train_frac
        split = int(len(df_raw) * frac)
        df_tr = df_raw.iloc[:split]
        self._split_date = df_tr.index[-1]

        n_latent = self._enc.latent_dim if self._enc else 0
        print(
            f"[Pipeline] build_features — {len(df_tr):,} train bars / "
            f"{len(df_raw):,} total  "
            f"(split at {self._split_date.date()})",
            flush=True,
        )

        # Base features — scaler fitted on train only
        _, _ = self._fp.build(df_tr, fit=True)           # fits scaler
        X_full, y_full = self._fp.build(df_raw, fit=False)  # transforms all

        # Latent encoder — trained on train only
        if self._enc is not None:
            y_for_enc = (
                y_full.reindex(df_tr.index)          # labels aligned to df_tr
                if self._enc.mode == "supervised" else None
            )
            print(
                f"[Pipeline] Training {self._enc.mode} encoder "
                f"(latent_dim={self._enc.latent_dim})...",
                flush=True,
            )
            self._enc.fit(df_tr, y=y_for_enc)

            latent_full = self._enc.transform(df_raw)      # (n_rows, latent_dim)

            # Align: X_full may be shorter (NaN rows + label_horizon dropped)
            shared   = X_full.index.intersection(latent_full.index)
            X_full   = pd.concat(
                [X_full.loc[shared], latent_full.loc[shared]], axis=1
            )
            y_full   = y_full.reindex(shared)

        y_full = y_full.dropna()
        X_full = X_full.reindex(y_full.index)

        self._feature_cols = list(X_full.columns)
        n_base = X_full.shape[1] - n_latent
        print(
            f"[Pipeline] Feature matrix: {X_full.shape}  "
            f"({n_base} base{f' + {n_latent} latent' if n_latent else ''})",
            flush=True,
        )
        return X_full, y_full

    # ── Step 2: Walk-forward evaluation ───────────────────────────────────────

    def walk_forward(
        self,
        X:      pd.DataFrame,
        y:      pd.Series,
        prices: pd.DataFrame,
    ) -> WalkForwardResult:
        """
        Run a walk-forward evaluation on pre-built feature matrix X, y.

        The model (XGBoost / LightGBM / etc.) is re-trained at every fold
        boundary on expanding or sliding training data.  The encoder and scaler
        are NOT re-fitted per fold (they were fitted in build_features).

        Returns WalkForwardResult with .report(), .equity, .trades, .folds.
        """
        cfg = self.cfg
        wf_cfg = WalkForwardConfig(
            model_type  = cfg.model_type,
            window_type = cfg.wf_window_type,
            train_days  = cfg.wf_train_days,
            test_days   = cfg.wf_test_days,
            backtest    = self._make_backtest_cfg(),
            cache_dir   = cfg.wf_cache_dir,
        )
        print(
            f"[Pipeline] walk_forward — model={cfg.model_type}  "
            f"window={cfg.wf_window_type}  "
            f"train={cfg.wf_train_days}d  test={cfg.wf_test_days}d",
            flush=True,
        )
        return WalkForwardValidator(verbose=True).run(X, y, prices, wf_cfg)

    # ── Step 3: Full-dataset training (for live deployment) ───────────────────

    def fit_full(self, X: pd.DataFrame, y: pd.Series) -> "PredictorPipeline":
        """
        Train the model on the complete feature matrix X.
        Call this after walk_forward() if results are satisfactory, before save().
        """
        print(
            f"[Pipeline] fit_full — {self.cfg.model_type} on "
            f"{len(X):,} rows × {X.shape[1]} features",
            flush=True,
        )
        self._model = _build_model(self.cfg.model_type)
        self._model.train(X, y)
        print("[Pipeline] fit_full complete.", flush=True)
        return self

    # ── Live inference ─────────────────────────────────────────────────────────

    def predict(self, df_raw: pd.DataFrame) -> dict:
        """
        Single-bar inference from raw OHLCV.

        Applies the fitted scaler → encoder → model and returns a signal dict.
        df_raw must contain at least max(indicator_warmup, encoder.window_size) bars.

        Returns
        -------
        {
            "signal"    : "buy" | "sell" | "hold",
            "confidence": float,
            "P_buy"     : float,
            "P_hold"    : float,
            "P_sell"    : float,
            "timestamp" : pd.Timestamp,
            "sizing"    : dict   — RiskManager output (risk_pct, skip, ...)
        }
        """
        if self._model is None:
            raise RuntimeError(
                "No model loaded. Call fit_full() or load() first."
            )

        # Base features for the last bar (scaler applied internally)
        X_live = self._fp.build_live(df_raw)

        # Append latent dims for the last bar
        if self._enc is not None:
            latent = self._enc.transform(df_raw)     # full series, same index
            lat_row = latent.iloc[[-1]].copy()
            lat_row.index = X_live.index
            X_live = pd.concat([X_live, lat_row], axis=1)

        # Align to training feature columns (fill any missing with 0)
        if self._feature_cols:
            for c in self._feature_cols:
                if c not in X_live.columns:
                    X_live[c] = 0.0
            X_live = X_live[self._feature_cols]

        proba = self._model.predict_proba(X_live)
        p = proba[-1] if proba.ndim == 2 else proba
        P_buy, P_hold, P_sell = float(p[0]), float(p[1]), float(p[2])
        confidence = max(P_buy, P_sell)
        thr = self.cfg.bt_threshold

        if P_buy >= thr and P_buy > P_sell:
            signal = "buy"
        elif P_sell >= thr and P_sell > P_buy:
            signal = "sell"
        else:
            signal = "hold"

        sizing = self._rm.size(
            confidence = confidence,
            balance    = self.cfg.bt_balance,
            sl_pips    = self.cfg.bt_sl_pips,
            tp_pips    = self.cfg.bt_tp_pips,
        )

        return {
            "signal":     signal,
            "confidence": round(confidence, 4),
            "P_buy":      round(P_buy,  4),
            "P_hold":     round(P_hold, 4),
            "P_sell":     round(P_sell, 4),
            "timestamp":  df_raw.index[-1],
            "sizing":     {
                "risk_pct": sizing.risk_pct,
                "skip":     sizing.skip,
                "sl_pips":  sizing.sl_pips,
            },
        }

    def predict_batch(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Batch inference on a pre-built feature matrix.
        Returns DataFrame with columns [P_buy, P_hold, P_sell, signal, confidence].
        """
        if self._model is None:
            raise RuntimeError("No model. Call fit_full() or load().")

        proba = self._model.predict_proba(X)
        if proba.ndim == 1:
            proba = proba.reshape(1, -1)

        df_out = pd.DataFrame(
            proba, index=X.index, columns=["P_buy", "P_hold", "P_sell"]
        )
        thr = self.cfg.bt_threshold
        df_out["confidence"] = df_out[["P_buy", "P_sell"]].max(axis=1)
        df_out["signal"] = "hold"
        df_out.loc[
            (df_out["P_buy"] >= thr) & (df_out["P_buy"] > df_out["P_sell"]), "signal"
        ] = "buy"
        df_out.loc[
            (df_out["P_sell"] >= thr) & (df_out["P_sell"] > df_out["P_buy"]), "signal"
        ] = "sell"
        return df_out

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, directory: str = None) -> None:
        """
        Save all fitted artifacts to directory.

        Writes:
          scaler.joblib    — StandardScaler from FeaturePipeline
          encoder.pt       — LatentEncoder weights (if enabled)
          model.joblib     — trained model (.pt for LSTM)
          meta.json        — config + feature column names
        """
        dir_path = Path(directory or self.cfg.artifacts_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        self._fp.save_scaler(dir_path / "scaler.joblib")

        if self._enc is not None and self._enc._net is not None:
            self._enc.save(str(dir_path / "encoder.pt"))

        if self._model is not None:
            ext = ".pt" if self.cfg.model_type == "lstm" else ".joblib"
            self._model.save(str(dir_path / f"model{ext}"))

        meta = {
            "model_type":      self.cfg.model_type,
            "encoder_enabled": self._enc is not None,
            "feature_cols":    self._feature_cols,
            "split_date":      str(self._split_date),
            "cfg":             asdict(self.cfg),
        }
        with open(dir_path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        print(
            f"[Pipeline] Saved → {dir_path}  "
            f"(model={self.cfg.model_type}  "
            f"features={len(self._feature_cols)}  "
            f"encoder={'yes' if self._enc else 'no'})",
            flush=True,
        )

    def load(self, directory: str = None) -> "PredictorPipeline":
        """
        Load all fitted artifacts from directory.
        After load(), predict() is ready to call.
        """
        dir_path = Path(directory or self.cfg.artifacts_dir)

        meta_path = dir_path / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No meta.json found in {dir_path}. Run 'train' mode first."
            )
        with open(meta_path) as f:
            meta = json.load(f)

        self._feature_cols = meta.get("feature_cols", [])
        self._split_date   = (
            pd.Timestamp(meta["split_date"]) if meta.get("split_date") else None
        )

        # Restore cfg from saved meta so predict() uses the right thresholds
        if "cfg" in meta:
            self.cfg = PipelineConfig.from_dict(meta["cfg"])

        # Restore FeaturePipeline's internal feature list so build_live() works.
        # base features = everything except latent_ columns added by the encoder.
        self._fp._feature_cols = [
            c for c in self._feature_cols if not c.startswith("latent_")
        ]

        # Scaler
        scaler_path = dir_path / "scaler.joblib"
        if scaler_path.exists():
            self._fp.load_scaler(scaler_path)

        # Encoder
        enc_path = dir_path / "encoder.pt"
        if meta.get("encoder_enabled") and enc_path.exists():
            cfg = self.cfg
            self._enc = LatentEncoder(
                mode        = cfg.encoder_mode,
                latent_dim  = cfg.encoder_latent_dim,
                window_size = cfg.encoder_window,
            )
            self._enc.load(str(enc_path))

        # Model
        ext        = ".pt" if meta["model_type"] == "lstm" else ".joblib"
        model_path = dir_path / f"model{ext}"
        if model_path.exists():
            self._model = _build_model(meta["model_type"])
            self._model.load(str(model_path))

        print(
            f"[Pipeline] Loaded ← {dir_path}  "
            f"(model={meta['model_type']}  features={len(self._feature_cols)})",
            flush=True,
        )
        return self

    # ── Inspection ─────────────────────────────────────────────────────────────

    def summary(self) -> None:
        """Print a human-readable summary of the pipeline state."""
        enc_meta = self._enc.metadata() if self._enc else {}
        print(self.cfg.summary())
        print(f"  fitted       : {'yes' if self._model else 'no'}")
        print(f"  feature_cols : {len(self._feature_cols)}"
              f"  ({self._feature_cols[:3]}...)" if self._feature_cols else "")
        if enc_meta:
            print(f"  enc_trained  : {enc_meta.get('fitted', False)}")

    def feature_names(self) -> list[str]:
        return list(self._feature_cols)

    # ── Private ────────────────────────────────────────────────────────────────

    def _build_rm(self) -> RiskManager:
        rc = RiskConfig()
        if self.cfg.rm_tiers:
            rc.tiers = [tuple(t) for t in self.cfg.rm_tiers]
        rc.min_confidence = self.cfg.bt_threshold
        return RiskManager(rc)

    def _make_backtest_cfg(self) -> BacktestConfig:
        return BacktestConfig(
            threshold         = self.cfg.bt_threshold,
            sl_pips           = self.cfg.bt_sl_pips,
            tp_pips           = self.cfg.bt_tp_pips,
            spread_pips       = self.cfg.bt_spread,
            initial_balance   = self.cfg.bt_balance,
            risk_pct          = self.cfg.bt_risk_pct,
            use_regime_filter = self.cfg.bt_use_regime,
            risk_manager      = self._rm,
        )
