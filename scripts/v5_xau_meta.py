"""V5 XAUUSD meta-labeling experiment — fold-local XGBoost (+ encoder) filter.

See the pre-registration block in V5_PLAN.MD ("XAUUSD Meta-Labeling
Experiment"). The validated engine generates the trades; XGBoost only
predicts per-trade win probability OOS and skips/resizes. Directional ML is
deliberately NOT attempted (failed strict validation historically).

    python scripts/v5_xau_meta.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cta.bootstrap import block_bootstrap_sharpe
from src.evaluation.metrics import max_drawdown
from src.v5.artifacts import V5ArtifactWriter
from src.v5.xau_trend import run_trades, wilder_atr, xau_signal

EVAL_START = "2018-01-01"          # first OOS test year
DATA = "data/XAUUSD_H4_long.csv"
CONF_RISK = {"low": 0.005, "med": 0.010, "high": 0.015}
SEED = 42

XGB_PARAMS = dict(max_depth=3, n_estimators=200, learning_rate=0.05,
                  subsample=0.8, random_state=SEED, n_jobs=4,
                  eval_metric="logloss")


def bar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Past-only per-bar features, SHIFTED so row t uses bars <= t-1."""
    close = df["close"]
    atr = wilder_atr(df, 14)
    sig = xau_signal(close)
    sma200 = close.rolling(200, min_periods=200).mean()
    feats = pd.DataFrame(index=df.index)
    feats["abs_forecast"] = sig.abs()
    feats["forecast_sign"] = np.sign(sig)
    feats["atr_over_price"] = atr / close
    feats["atr_pctile_1y"] = (atr.rolling(1560, min_periods=390)
                              .rank(pct=True))
    for n in (5, 21, 63):
        feats[f"ret_{n}_atr"] = (close - close.shift(n)) / (atr * np.sqrt(n))
    feats["dist_sma200_atr"] = (close - sma200) / atr
    feats["spread"] = df["spread"]
    shifted = feats.shift(1)  # decision-bar information only
    shifted["dow"] = df.index.dayofweek
    shifted["hour"] = df.index.hour
    return shifted


class TinyAutoencoder:
    """Fold-local 8-dim autoencoder on the bar-feature matrix (fixed seed)."""

    def __init__(self, latent: int = 8, epochs: int = 20):
        self.latent, self.epochs = latent, epochs
        self.net, self.mu, self.sd = None, None, None

    def fit(self, x: np.ndarray):
        import torch
        from torch import nn
        torch.manual_seed(SEED)
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-9
        xt = torch.tensor((x - self.mu) / self.sd, dtype=torch.float32)
        d = x.shape[1]
        self.net = nn.Sequential(nn.Linear(d, 32), nn.ReLU(),
                                 nn.Linear(32, self.latent))
        dec = nn.Sequential(nn.ReLU(), nn.Linear(self.latent, 32), nn.ReLU(),
                            nn.Linear(32, d))
        opt = torch.optim.Adam(list(self.net.parameters()) + list(dec.parameters()),
                               lr=1e-3)
        loss_fn = nn.MSELoss()
        for _ in range(self.epochs):
            opt.zero_grad()
            out = dec(self.net(xt))
            loss = loss_fn(out, xt)
            loss.backward()
            opt.step()
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            xt = torch.tensor((x - self.mu) / self.sd, dtype=torch.float32)
            return self.net(xt).numpy()


def trade_table(df: pd.DataFrame) -> pd.DataFrame:
    res = run_trades(df, exit_mode="trail", flip_mode="confidence",
                     params={"conf_risk_scale": {"low": 0.5, "med": 1.0, "high": 1.5}})
    t = res["trades"].copy()
    t["open_time"] = pd.to_datetime(t["open_time"])
    t["close_time"] = pd.to_datetime(t["close_time"])
    t["risk_frac"] = t["confidence"].map(CONF_RISK)
    t["y"] = (t["r_multiple"] > 0).astype(int)
    return t.dropna(subset=["r_multiple"])


def rolling_prior_r(trades: pd.DataFrame, n: int = 5) -> pd.Series:
    return (trades["r_multiple"].rolling(n, min_periods=1).mean()
            .shift(1).fillna(0.0))


def oos_probabilities(trades: pd.DataFrame, feats: pd.DataFrame,
                      bar_matrix: pd.DataFrame, use_encoder: bool) -> pd.Series:
    """Expanding yearly folds; purge = train on trades CLOSED before test."""
    from xgboost import XGBClassifier
    x_trades = feats.loc[trades["open_time"]].reset_index(drop=True)
    x_trades["prior_r5"] = rolling_prior_r(trades).values
    probs = pd.Series(np.nan, index=trades.index)
    fold_aucs = []
    years = range(2018, trades["close_time"].dt.year.max() + 1)
    for year in years:
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year + 1}-01-01")
        train_mask = trades["close_time"] < test_start
        test_mask = (trades["open_time"] >= test_start) & \
                    (trades["open_time"] < test_end)
        if train_mask.sum() < 100 or test_mask.sum() == 0:
            continue
        xtr = x_trades[train_mask.values].values.astype(float)
        xte = x_trades[test_mask.values].values.astype(float)
        if use_encoder:
            bars_train = bar_matrix[bar_matrix.index < test_start].dropna()
            enc = TinyAutoencoder().fit(bars_train.values.astype(float))
            xtr = np.hstack([xtr, enc.transform(
                feats.loc[trades.loc[train_mask, "open_time"]].values.astype(float))])
            xte = np.hstack([xte, enc.transform(
                feats.loc[trades.loc[test_mask, "open_time"]].values.astype(float))])
        m = XGBClassifier(**XGB_PARAMS)
        m.fit(np.nan_to_num(xtr), trades.loc[train_mask, "y"].values)
        p = m.predict_proba(np.nan_to_num(xte))[:, 1]
        probs[test_mask.values] = p
        yte = trades.loc[test_mask, "y"].values
        if 0 < yte.sum() < len(yte):
            from sklearn.metrics import roc_auc_score
            fold_aucs.append(roc_auc_score(yte, p))
    print(f"  fold OOS AUCs ({'enc' if use_encoder else 'xgb'}): "
          f"{[round(a, 3) for a in fold_aucs]}  mean {np.mean(fold_aucs):.3f}")
    return probs, float(np.mean(fold_aucs))


def equity_from_trades(trades: pd.DataFrame, risk_mult: pd.Series) -> pd.Series:
    """Compound equity from R x risk_frac x mult, marked at close times."""
    ret = trades["r_multiple"] * trades["risk_frac"] * risk_mult
    eq = (1.0 + ret).cumprod()
    return pd.Series(eq.values, index=trades["close_time"].values).groupby(level=0).last()


def evaluate(trades: pd.DataFrame, risk_mult: pd.Series, label: str) -> dict:
    sel = trades["open_time"] >= EVAL_START
    t = trades[sel]
    eq = equity_from_trades(t, risk_mult[sel])
    daily = eq.resample("D").last().ffill().pct_change(fill_method=None).dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    ci = block_bootstrap_sharpe(daily.values)
    active = t[risk_mult[sel].values > 0]
    stats = dict(variant=label,
                 sharpe=round(sharpe, 3),
                 sharpe_ci95=[round(ci[0], 3), round(ci[1], 3)],
                 total_return_pct=round((eq.iloc[-1] - 1) * 100, 1),
                 max_dd_pct=round(max_drawdown(eq), 2),
                 n_trades=int(len(t)), n_taken=int(len(active)),
                 win_rate_pct=round((active["r_multiple"] > 0).mean() * 100, 1)
                 if len(active) else 0.0)
    print(f"  {label:22s} Sharpe {stats['sharpe']:+.3f} CI {stats['sharpe_ci95']} "
          f"ret {stats['total_return_pct']:+.1f}% DD {stats['max_dd_pct']:.1f}% "
          f"taken {stats['n_taken']}/{stats['n_trades']}")
    return stats


def main() -> None:
    df = pd.read_csv(DATA, parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    feats = bar_features(df)
    trades = trade_table(df).reset_index(drop=True)
    print(f"engine trades: {len(trades)} "
          f"({trades['open_time'].min().date()} -> {trades['close_time'].max().date()})")

    ones = pd.Series(1.0, index=trades.index)
    results = [evaluate(trades, ones, "baseline")]

    p_xgb, auc_xgb = oos_probabilities(trades, feats, feats, use_encoder=False)
    valid = p_xgb.notna()
    skip = ones.where(~valid, (p_xgb >= 0.4).astype(float))
    results.append(evaluate(trades, skip, "A_skip_p<0.4"))
    scale = ones.where(~valid, ((p_xgb - 0.2) / 0.6).clip(0.5, 1.5))
    results.append(evaluate(trades, scale, "B_risk_x_p"))

    p_enc, auc_enc = oos_probabilities(trades, feats, feats, use_encoder=True)
    valid_e = p_enc.notna()
    scale_e = ones.where(~valid_e, ((p_enc - 0.2) / 0.6).clip(0.5, 1.5))
    results.append(evaluate(trades, scale_e, "C_risk_x_p_enc8"))

    stats = {"results": results,
             "oos_auc": {"xgb": round(auc_xgb, 3), "xgb_enc8": round(auc_enc, 3)},
             "note": "meta-labeling on the promoted engine's own trades; "
                     "identical R-based equity metric for all rows"}
    eq = equity_from_trades(trades[trades["open_time"] >= EVAL_START],
                            ones[trades["open_time"] >= EVAL_START])
    V5ArtifactWriter().write_run(
        run_id="xau-meta-xgb", settings={"strategy": "xau_meta_labeling",
                                         "pre_registration": "V5_PLAN.MD",
                                         "xgb_params": XGB_PARAMS},
        trades=trades.to_dict("records"), equity=eq, stats=stats,
        reconciliation={"status": "research_replay"})
    print("run_dir: data/v5_runs/xau-meta-xgb")


if __name__ == "__main__":
    main()
