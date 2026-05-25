"""
Walk-Forward Validation Engine — Phase 7.

A proper walk-forward engine that:
  - Supports both expanding and sliding training windows
  - Retrains any ModelInterface (including Ensemble) at each fold boundary
  - Caches trained models to disk so expensive folds aren't re-run unnecessarily
  - Uses Backtester for trade simulation (unified cost model)
  - Aggregates per-fold equity curves into a single out-of-sample equity curve

This replaces / supersedes scripts/walk_forward.py which was the Phase 4/5
prototype.  The prototype is kept for backward compatibility.

Usage:
    from src.walk_forward import WalkForwardValidator, WalkForwardConfig
    from src.backtester  import BacktestConfig

    cfg = WalkForwardConfig(
        model_type="xgboost",
        window_type="expanding",
        train_days=180,
        test_days=30,
        backtest=BacktestConfig(threshold=0.40, use_regime_filter=True),
    )
    result = WalkForwardValidator().run(X, y, prices, cfg)
    result.report()
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.backtester import Backtester, BacktestConfig, BacktestResult
from src.metrics import performance_report, sharpe_ratio, max_drawdown


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class WalkForwardConfig:
    model_type:  str = "xgboost"         # xgboost | lightgbm | catboost | random_forest | ensemble
    window_type: str = "expanding"       # "expanding" | "sliding"
    train_days:  int = 180
    test_days:   int = 30

    # Ensemble-only options (ignored for single models)
    ensemble_base_models: list = field(default_factory=lambda: [
        "xgboost", "lightgbm", "catboost", "random_forest"
    ])
    ensemble_meta_model: str = "lightgbm"
    ensemble_n_folds:    int = 5

    # Caching: save trained model per fold to avoid re-training
    cache_dir: Optional[str] = "data/models/wf_cache"

    # Passed through to Backtester
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


# ── Per-fold result ─────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    fold:         int
    train_start:  object
    train_end:    object
    test_start:   object
    test_end:     object
    n_trades:     int
    win_rate:     float
    sharpe:       float
    total_return: float
    n_train_bars: int
    n_test_bars:  int


# ── Aggregate result ───────────────────────────────────────────────────────────

@dataclass
class WalkForwardResult:
    trades:  list[dict]
    equity:  pd.Series
    folds:   list[FoldResult]
    config:  WalkForwardConfig

    @property
    def sharpe(self) -> float:
        return sharpe_ratio(self.equity)

    @property
    def drawdown(self) -> float:
        return max_drawdown(self.equity)

    def print_fold_table(self) -> None:
        w = 84
        print("─" * w)
        print(f"{'Fold':>4}  {'Train window':>27}  {'Test window':>22}  "
              f"{'Trades':>6}  {'WinRate':>7}  {'Sharpe':>6}  {'Return':>7}")
        print("─" * w)
        for f in self.folds:
            tw = f"{f.train_start} → {f.train_end}"
            vw = f"{f.test_start} → {f.test_end}"
            print(f"  {f.fold:>2}  {tw:>27}  {vw:>22}  "
                  f"{f.n_trades:>6}  {f.win_rate*100:>6.1f}%  "
                  f"{f.sharpe:>6.2f}  {f.total_return:>+6.1f}%")
        print("─" * w)

    def report(self, title: Optional[str] = None) -> None:
        cfg = self.config
        t   = title or f"WALK-FORWARD ({cfg.model_type.upper()})"
        self.print_fold_table()
        if not self.trades or len(self.equity) == 0:
            print("No trades generated — try lowering threshold.")
            return
        performance_report(
            self.trades, self.equity,
            cfg.backtest.initial_balance,
            title=t,
            extra_params={
                "Model":         cfg.model_type,
                "Window":        cfg.window_type,
                "Train / Test":  f"{cfg.train_days}d / {cfg.test_days}d",
                "Folds":         len(self.folds),
                "Threshold":     f"{cfg.backtest.threshold:.0%}",
                "SL / TP":       f"{cfg.backtest.sl_pips}p / {cfg.backtest.tp_pips}p",
                "Spread":        f"{cfg.backtest.spread_pips}p",
                "Regime filter": ("ON" if cfg.backtest.use_regime_filter else "OFF"),
            },
        )


# ── Validator ──────────────────────────────────────────────────────────────────

class WalkForwardValidator:
    """
    Orchestrates a walk-forward validation run.

    Parameters
    ----------
    verbose : bool
        Print fold-level progress during training.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        X:      pd.DataFrame,
        y:      pd.Series,
        prices: pd.DataFrame,
        config: WalkForwardConfig,
    ) -> WalkForwardResult:
        """
        Run the full walk-forward validation.

        X      — feature matrix (DatetimeIndex)
        y      — labels aligned to X
        prices — OHLCV DataFrame (DatetimeIndex)
        config — WalkForwardConfig
        """
        if config.cache_dir:
            Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

        bt          = Backtester()
        all_trades  = []
        equity_segs = []
        fold_results= []
        balance     = config.backtest.initial_balance
        fold        = 0

        dates    = X.index
        start_dt = dates[0]
        train_end = start_dt + pd.Timedelta(days=config.train_days)

        while train_end < dates[-1]:
            test_end = min(
                train_end + pd.Timedelta(days=config.test_days),
                dates[-1],
            )

            # Slice
            if config.window_type == "sliding":
                train_start = train_end - pd.Timedelta(days=config.train_days)
                X_train = X[(X.index >= train_start) & (X.index < train_end)]
                y_train = y[(y.index >= train_start) & (y.index < train_end)]
            else:  # expanding
                X_train = X[X.index < train_end]
                y_train = y[y.index < train_end]

            X_test = X[(X.index >= train_end) & (X.index < test_end)]

            if len(X_train) < 500 or len(X_test) < 10:
                train_end = test_end
                continue

            if self.verbose:
                mode = "expanding" if config.window_type == "expanding" else "sliding"
                print(f"  Fold {fold}: train {len(X_train):,} bars [{mode}]  "
                      f"test {len(X_test):,} bars")

            # Build / load model
            model = self._get_model(X_train, y_train, config, fold)

            # Carry current balance into this fold's backtester config
            fold_cfg = BacktestConfig(
                threshold=config.backtest.threshold,
                pip_size=config.backtest.pip_size,
                sl_pips=config.backtest.sl_pips,
                tp_pips=config.backtest.tp_pips,
                spread_pips=config.backtest.spread_pips,
                commission_pips=config.backtest.commission_pips,
                max_slippage_pips=config.backtest.max_slippage_pips,
                initial_balance=balance,
                risk_pct=config.backtest.risk_pct,
                use_regime_filter=config.backtest.use_regime_filter,
                adx_threshold=config.backtest.adx_threshold,
                adx_col=config.backtest.adx_col,
            )

            result: BacktestResult = bt.run(model, X_test, prices, fold_cfg, fold=fold)

            # Update running balance from last equity point
            if len(result.equity) > 0:
                balance = float(result.equity.iloc[-1])
                # Add any P&L from force-closed trade at end
                for t in result.trades:
                    pass  # balance already baked into equity

            all_trades.extend(result.trades)
            if len(result.equity) > 0:
                equity_segs.append(result.equity)

            # Per-fold stats
            pnl     = [t["pnl_pips"] for t in result.trades]
            wins    = [p for p in pnl if p > 0]
            wr      = len(wins) / len(pnl) if pnl else 0.0
            f_sharpe= sharpe_ratio(result.equity) if len(result.equity) > 5 else 0.0
            ret     = (
                (result.equity.iloc[-1] / result.equity.iloc[0] - 1) * 100
                if len(result.equity) > 0 else 0.0
            )

            fold_results.append(FoldResult(
                fold=fold,
                train_start=X_train.index[0].date(),
                train_end=train_end.date(),
                test_start=train_end.date(),
                test_end=test_end.date(),
                n_trades=len(result.trades),
                win_rate=wr,
                sharpe=f_sharpe,
                total_return=ret,
                n_train_bars=len(X_train),
                n_test_bars=len(X_test),
            ))

            fold      += 1
            train_end  = test_end   # always expand forward

        equity = pd.concat(equity_segs) if equity_segs else pd.Series(dtype=float)
        return WalkForwardResult(
            trades=all_trades,
            equity=equity,
            folds=fold_results,
            config=config,
        )

    # ── Model building ─────────────────────────────────────────────────────────

    def _get_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        config:  WalkForwardConfig,
        fold:    int,
    ):
        """Train (or load from cache) a model for this fold."""
        cache_path = self._cache_path(config, fold, X_train)

        if cache_path and cache_path.exists():
            if self.verbose:
                print(f"    Loading cached model: {cache_path.name}")
            return self._load_cached(cache_path, config.model_type)

        model = self._build_and_train(X_train, y_train, config)

        if cache_path:
            if self.verbose:
                print(f"    Saving model cache: {cache_path.name}")
            self._save_cached(model, cache_path, config.model_type)

        return model

    def _build_and_train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        config:  WalkForwardConfig,
    ):
        mt = config.model_type.lower()
        if mt == "ensemble":
            return self._build_ensemble(X_train, y_train, config)

        from src.model_registry import _build_model
        model = _build_model(mt)
        model.train(X_train, y_train)
        return model

    def _build_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        config:  WalkForwardConfig,
    ):
        from src.ensemble import Ensemble
        from src.model_registry import _build_model

        base_models = [_build_model(n) for n in config.ensemble_base_models]
        ens = Ensemble(
            base_models=base_models,
            meta_model=config.ensemble_meta_model,
            n_folds=config.ensemble_n_folds,
        )
        ens.train(X_train, y_train)
        return ens

    # ── Caching ────────────────────────────────────────────────────────────────

    def _cache_path(
        self,
        config: WalkForwardConfig,
        fold:   int,
        X_train: pd.DataFrame,
    ) -> Optional[Path]:
        if not config.cache_dir:
            return None
        # Cache key: model_type + fold + training window end date
        key_data = {
            "model":      config.model_type,
            "fold":       fold,
            "train_end":  str(X_train.index[-1]),
            "window":     config.window_type,
            "train_days": config.train_days,
        }
        key_hash = hashlib.md5(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()[:10]
        suffix = ".joblib" if config.model_type != "lstm" else ".pt"
        return Path(config.cache_dir) / f"{config.model_type}_fold{fold}_{key_hash}{suffix}"

    def _save_cached(self, model, path: Path, model_type: str) -> None:
        try:
            model.save(str(path))
        except Exception:
            pass  # cache failure is non-fatal

    def _load_cached(self, path: Path, model_type: str):
        from src.model_registry import _build_model
        mt = model_type.lower()
        if mt == "ensemble":
            from src.ensemble import Ensemble
            ens = Ensemble(base_models=[])
            ens.load(str(path))
            return ens
        model = _build_model(mt)
        model.load(str(path))
        return model
