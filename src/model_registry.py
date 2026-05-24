"""
Model Registry — Phase 5.

Runtime registry for swapping prediction models without code changes.
The active model is determined by config.yaml: active_model.

Usage:
    from src.model_registry import ModelRegistry

    registry = ModelRegistry()

    # Register models
    registry.register("xgboost",      XGBoostModel().load("data/models/xgboost.joblib"))
    registry.register("lightgbm",     LightGBMModel().load("data/models/lightgbm.joblib"))
    registry.register("random_forest", RandomForestModel().load("data/models/rf.joblib"))

    # Set active model (also done automatically from config.yaml)
    registry.set_active("lightgbm")

    # Get the active model — the trading engine never changes
    proba = registry.get_active().predict_proba(features)

    # Inspect
    registry.list_models()
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import yaml

from src.model_interface import ModelInterface


class ModelRegistry:
    """
    Singleton-style registry that maps model names to ModelInterface instances.

    The active model is the one the trading engine calls. Swap it at runtime
    via set_active() or from a config.yaml change + reload.
    """

    _instance: Optional["ModelRegistry"] = None

    def __new__(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models: Dict[str, ModelInterface] = {}
            cls._instance._active: Optional[str] = None
        return cls._instance

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, name: str, model: ModelInterface) -> "ModelRegistry":
        """Add or replace a model. Returns self for chaining."""
        if not isinstance(model, ModelInterface):
            raise TypeError(f"Model must implement ModelInterface, got {type(model)}")
        self._models[name] = model
        if self._active is None:
            self._active = name
        return self

    def unregister(self, name: str) -> None:
        if name not in self._models:
            raise KeyError(f"No model named '{name}' in registry.")
        del self._models[name]
        if self._active == name:
            self._active = next(iter(self._models), None)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get(self, name: str) -> ModelInterface:
        if name not in self._models:
            raise KeyError(f"No model named '{name}'. Registered: {list(self._models)}")
        return self._models[name]

    def get_active(self) -> ModelInterface:
        if self._active is None or self._active not in self._models:
            raise RuntimeError(
                "No active model set. Call register() or set_active() first."
            )
        return self._models[self._active]

    def set_active(self, name: str) -> "ModelRegistry":
        if name not in self._models:
            raise KeyError(f"Cannot set active: no model named '{name}'.")
        self._active = name
        return self

    @property
    def active_name(self) -> Optional[str]:
        return self._active

    # ── Introspection ─────────────────────────────────────────────────────────

    def list_models(self) -> None:
        """Print a formatted table of all registered models with their metadata."""
        if not self._models:
            print("Registry is empty.")
            return
        w = 70
        print("─" * w)
        print(f"{'Name':<20}  {'Active':^6}  {'Trained on':<25}  {'Features'}")
        print("─" * w)
        for name, model in self._models.items():
            try:
                meta = model.metadata()
                trained = meta.get("trained_on", "—")
                n_feat  = len(meta.get("features", []))
            except Exception:
                trained, n_feat = "—", "—"
            active_marker = "  ★" if name == self._active else ""
            print(f"  {name:<18}  {'yes' if name == self._active else 'no':^6}  "
                  f"{str(trained):<25}  {n_feat}{active_marker}")
        print("─" * w)

    def names(self) -> list[str]:
        return list(self._models.keys())

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, name: str) -> bool:
        return name in self._models

    # ── Config integration ────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "config.yaml",
        auto_load: bool = True,
    ) -> "ModelRegistry":
        """
        Build a registry from config.yaml.

        Reads config.yaml for the 'models' section, instantiates each model,
        and sets the active model from 'active_model'.

        If auto_load=True (default), each model is loaded from its artifact path.
        If auto_load=False, models are registered but not loaded (useful for training).

        Config format:
            active_model: xgboost
            models:
              xgboost:
                type: xgboost
                path: data/models/xgboost.joblib
              lightgbm:
                type: lightgbm
                path: data/models/lightgbm.joblib
              random_forest:
                type: random_forest
                path: data/models/random_forest.joblib
        """
        registry = cls()

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        models_cfg  = cfg.get("models", {})
        active_name = cfg.get("active_model", None)

        for name, spec in models_cfg.items():
            model = _build_model(spec["type"])
            if auto_load:
                path = Path(spec.get("path", f"data/models/{name}.joblib"))
                if path.exists():
                    model.load(path)
                    registry.register(name, model)
                else:
                    print(f"[Registry] Skipping '{name}': artifact not found at {path}")
            else:
                registry.register(name, model)

        if active_name and active_name in registry:
            registry.set_active(active_name)
        elif active_name:
            print(f"[Registry] Warning: active_model='{active_name}' not loaded — "
                  f"using '{registry.active_name}'")

        return registry

    @classmethod
    def reset(cls) -> None:
        """Clear the singleton (useful for tests)."""
        cls._instance = None


def _build_model(model_type: str) -> ModelInterface:
    """Instantiate a model by type string."""
    t = model_type.lower().replace("-", "_").replace(" ", "_")
    if t in ("xgboost", "xgb"):
        from src.models.xgboost_model import XGBoostModel
        return XGBoostModel()
    if t in ("lightgbm", "lgbm"):
        from src.models.lightgbm_model import LightGBMModel
        return LightGBMModel()
    if t in ("random_forest", "rf", "randomforest"):
        from src.models.random_forest_model import RandomForestModel
        return RandomForestModel()
    raise ValueError(f"Unknown model type: '{model_type}'. "
                     f"Choose from: xgboost, lightgbm, random_forest")
