"""V5 strict champion validation runner.

This ties together strict fold-local signals, broker-realistic replay, and
Lumibot-style artifacts. It is intentionally research-only until paper/live
reconciliation exists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from src.pipeline import PipelineConfig
from src.v5.artifacts import V5ArtifactWriter
from src.v5.replay import V5ReplayResult, replay_signal_frame
from src.v5.strict_pipeline import StrictWalkForwardResult, run_strict_walk_forward
from src.v5.validation import BrokerExecutionRules


@dataclass
class V5ChampionValidationConfig:
    symbol: str
    data_path: str | Path
    run_id: str
    artifact_root: str | Path
    requested_lot: float
    sl_pips: float
    tp_pips: float
    pipeline: PipelineConfig
    broker_rules: BrokerExecutionRules
    candle_features_path: str | Path | None = None
    initial_balance: float = 10_000.0
    max_folds: int | None = None


@dataclass
class V5ChampionValidationResult:
    run_dir: Path
    strict: StrictWalkForwardResult
    replay: V5ReplayResult
    stats: dict


def run_champion_validation(
    cfg: V5ChampionValidationConfig,
    *,
    model_factory: Callable[[str], object] | None = None,
) -> V5ChampionValidationResult:
    raw = load_ohlcv(cfg.data_path)
    candle_features = (
        load_candle_features(cfg.candle_features_path)
        if cfg.candle_features_path is not None
        else None
    )
    strict = run_strict_walk_forward(
        raw,
        cfg.pipeline,
        model_factory=model_factory,
        oos_candle_features=candle_features,
        max_folds=cfg.max_folds,
    )
    replay_signals = strict.signals.copy()
    if len(replay_signals) > 0:
        replay_signals["requested_lot"] = cfg.requested_lot
    replay = replay_signal_frame(
        raw,
        replay_signals,
        cfg.broker_rules,
        sl_pips=cfg.sl_pips,
        tp_pips=cfg.tp_pips,
        initial_balance=cfg.initial_balance,
    )
    stats = _stats(cfg, strict, replay)
    run_dir = V5ArtifactWriter(cfg.artifact_root).write_run(
        run_id=cfg.run_id,
        settings=_settings(cfg),
        trades=replay.trades,
        equity=replay.equity,
        stats=stats,
        folds=_fold_rows(strict),
        reconciliation={
            "status": "research_replay_only",
            "note": "Strict fold-local replay; not yet paper/live reconciled.",
        },
    )
    return V5ChampionValidationResult(
        run_dir=run_dir,
        strict=strict,
        replay=replay,
        stats=stats,
    )


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [c.lower() for c in frame.columns]
    time_col = next((c for c in frame.columns if "time" in c or c in {"date", "datetime"}), None)
    if time_col is not None:
        frame[time_col] = pd.to_datetime(frame[time_col])
        frame = frame.set_index(time_col)
    else:
        frame.index = pd.to_datetime(frame.index)
    return frame.sort_index()


def load_candle_features(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    frame.columns = [c.lower() for c in frame.columns]
    time_col = next((c for c in frame.columns if "time" in c or c in {"date", "datetime"}), None)
    if time_col is not None:
        frame[time_col] = pd.to_datetime(frame[time_col])
        frame = frame.set_index(time_col)
    else:
        frame.index = pd.to_datetime(frame.index)
    rename = {
        "p_buy": "candle_p_buy",
        "p_sell": "candle_p_sell",
        "p_hold": "candle_p_hold",
    }
    frame = frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns})
    return frame.sort_index()


def default_broker_rules_for_symbol(symbol: str) -> BrokerExecutionRules:
    pip_size = 0.01 if symbol.upper().endswith("JPY") or symbol.upper() == "XAUUSD" else 0.0001
    return BrokerExecutionRules(
        pip_size=pip_size,
        spread_pips=1.0,
        commission_pips=0.5,
        slippage_pips=0.3,
        entry_delay_bars=1,
        min_lot=0.01,
        lot_step=0.01,
        max_lot=0.50,
    )


def _settings(cfg: V5ChampionValidationConfig) -> dict:
    return {
        "symbol": cfg.symbol,
        "data_path": str(cfg.data_path),
        "requested_lot": cfg.requested_lot,
        "sl_pips": cfg.sl_pips,
        "tp_pips": cfg.tp_pips,
        "initial_balance": cfg.initial_balance,
        "pipeline": asdict(cfg.pipeline),
        "broker_rules": asdict(cfg.broker_rules),
        "candle_features_path": str(cfg.candle_features_path) if cfg.candle_features_path else None,
        "max_folds": cfg.max_folds,
    }


def _fold_rows(strict: StrictWalkForwardResult) -> list[dict]:
    by_fold = {fold.window.fold: fold for fold in strict.folds}
    rows = []
    for record in strict.fit_records:
        row = dict(record)
        fold = by_fold.get(record["fold"])
        if fold is not None:
            row["n_train_rows"] = fold.n_train_rows
            row["n_test_rows"] = fold.n_test_rows
        rows.append(row)
    return rows


def _stats(
    cfg: V5ChampionValidationConfig,
    strict: StrictWalkForwardResult,
    replay: V5ReplayResult,
) -> dict:
    equity = replay.equity
    total_return = (
        float(equity.iloc[-1] / equity.iloc[0] - 1.0)
        if len(equity) > 1 and equity.iloc[0] else 0.0
    )
    drawdown = _max_drawdown(equity)
    pnl = [float(t.get("pnl_pips", 0.0)) for t in replay.trades]
    wins = [x for x in pnl if x > 0]
    return {
        "symbol": cfg.symbol,
        "folds": len(strict.folds),
        "signals": len(strict.signals),
        "trades": len(replay.trades),
        "win_rate": len(wins) / len(pnl) if pnl else 0.0,
        "total_return": total_return,
        "max_drawdown": drawdown,
        "final_equity": float(equity.iloc[-1]) if len(equity) else cfg.initial_balance,
        "research_only": True,
    }


def _max_drawdown(equity: pd.Series) -> float:
    if len(equity) == 0:
        return 0.0
    peak = equity.cummax()
    dd = (peak - equity) / peak.replace(0, pd.NA)
    return float(dd.fillna(0.0).max())
