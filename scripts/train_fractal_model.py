"""
Fractal Pattern Model Training — based on READ 7.

Trains two CatBoost models using fractal-symmetry labels:
  1. Direction model  — predicts BUY (0) vs SELL (1) for bars inside fractals
  2. Meta model       — predicts TRADE (1) vs SKIP (0) for all bars

Compares against the standard forward-return labeling used by FeaturePipeline.

Usage
-----
    conda run -n envmt5 python scripts/train_fractal_model.py
    conda run -n envmt5 python scripts/train_fractal_model.py --symbol USDJPY
    conda run -n envmt5 python scripts/train_fractal_model.py --corr-threshold 0.7 --horizon 10
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features.feature_pipeline import FeaturePipeline
from src.features.fractal_labeler  import FractalLabeler


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data",           default=None,  help="Path to OHLCV CSV")
    p.add_argument("--symbol",         default="EURUSD")
    p.add_argument("--train-frac",     type=float, default=0.80)
    p.add_argument("--min-window",     type=int,   default=6)
    p.add_argument("--max-window",     type=int,   default=60)
    p.add_argument("--corr-threshold", type=float, default=0.9)
    p.add_argument("--horizon",        type=int,   default=5)
    p.add_argument("--markup",         type=float, default=0.0001)
    p.add_argument("--save",           action="store_true",
                   help="Save trained models to data/models/fractal_{symbol}/")
    return p.parse_args()


# ── Data helpers ───────────────────────────────────────────────────────────────

def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    time_col = next(c for c in df.columns if "time" in c)
    df[time_col] = pd.to_datetime(df[time_col])
    return df.set_index(time_col).sort_index()


def _train_val_split(df: pd.DataFrame, train_frac: float):
    n = len(df)
    cut = int(n * train_frac)
    return df.iloc[:cut], df.iloc[cut:]


# ── Metrics ────────────────────────────────────────────────────────────────────

def _accuracy(y_true, y_pred) -> float:
    return float((y_true == y_pred).mean())


def _f1_binary(y_true, y_pred, pos_label=1) -> float:
    tp = ((y_pred == pos_label) & (y_true == pos_label)).sum()
    fp = ((y_pred == pos_label) & (y_true != pos_label)).sum()
    fn = ((y_pred != pos_label) & (y_true == pos_label)).sum()
    p  = tp / max(tp + fp, 1)
    r  = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-10)


def _print_sep(title: str = "", char: str = "─", width: int = 60) -> None:
    if title:
        pad = (width - len(title) - 2) // 2
        print(char * pad + f" {title} " + char * pad)
    else:
        print(char * width)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        print("CatBoost not installed. Run: pip install catboost")
        sys.exit(1)

    args = _parse_args()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    data_path = args.data or f"data/{args.symbol}_M15.csv"
    print(f"\nLoading {data_path} …")
    df_raw = _load_csv(data_path)
    print(f"  {len(df_raw):,} bars  {df_raw.index[0].date()} → {df_raw.index[-1].date()}")

    df_train_raw, df_val_raw = _train_val_split(df_raw, args.train_frac)
    print(f"  Train: {len(df_train_raw):,}  Val: {len(df_val_raw):,}")

    # ── 2. Build features (no lookahead) ──────────────────────────────────────
    print("\nBuilding features …")
    fp = FeaturePipeline(scale=True)
    X_train_std, y_fwd_train = fp.build(df_train_raw, fit=True)
    X_val_std,   y_fwd_val   = fp.build(df_val_raw,   fit=False)

    # Add article-specific features: rolling std at periods [15,45,75,...,285]
    # These are the multi-scale volatility fingerprints the article uses to
    # describe fractal pattern shape (our standard TA features don't include them).
    _STD_PERIODS = list(range(15, 286, 30))  # [15, 45, 75, 105, ..., 285]

    def _add_fractal_features(df_raw: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        close = df_raw["close"]
        extra_cols = {}
        for p in _STD_PERIODS:
            col = f"frac_std_{p}"
            # Shift by 1 to avoid lookahead, align to X index
            s = close.rolling(p).std().shift(1).reindex(X.index)
            extra_cols[col] = s
        extra = pd.DataFrame(extra_cols, index=X.index)
        # Scale each column using stats from training data only
        return extra

    extra_tr = _add_fractal_features(df_train_raw, X_train_std)
    extra_vl = _add_fractal_features(df_val_raw,   X_val_std)

    # Fit a separate scaler on the extra cols (train only)
    _extra_scaler = StandardScaler()
    extra_tr_scaled = pd.DataFrame(
        _extra_scaler.fit_transform(extra_tr.fillna(0)),
        index=extra_tr.index, columns=extra_tr.columns,
    )
    extra_vl_scaled = pd.DataFrame(
        _extra_scaler.transform(extra_vl.fillna(0)),
        index=extra_vl.index, columns=extra_vl.columns,
    )
    X_train_std = pd.concat([X_train_std, extra_tr_scaled], axis=1)
    X_val_std   = pd.concat([X_val_std,   extra_vl_scaled], axis=1)

    print(f"  Features: {X_train_std.shape[1]} cols (incl. {len(_STD_PERIODS)} fractal-std)  "
          f"Train rows: {len(X_train_std):,}  Val rows: {len(X_val_std):,}")

    # ── 3. Generate fractal labels ────────────────────────────────────────────
    print("\nGenerating fractal labels …")
    labeler = FractalLabeler(
        min_window     = args.min_window,
        max_window     = args.max_window,
        corr_threshold = args.corr_threshold,
        horizon        = args.horizon,
        markup         = args.markup,
    )

    y_frac_full_train = labeler.label(df_train_raw)
    y_frac_full_val   = labeler.label(df_val_raw)

    # Align to feature matrix index (feature builder drops warmup rows)
    y_frac_train = y_frac_full_train.reindex(X_train_std.index)
    y_frac_val   = y_frac_full_val.reindex(X_val_std.index)

    # Drop any rows where fractal label is NaN after reindex
    valid_train = ~y_frac_train.isna()
    valid_val   = ~y_frac_val.isna()
    X_train_std = X_train_std[valid_train]
    y_frac_train = y_frac_train[valid_train].astype(int)
    X_val_std    = X_val_std[valid_val]
    y_frac_val   = y_frac_val[valid_val].astype(int)
    y_fwd_train  = y_fwd_train.reindex(X_train_std.index)
    y_fwd_val    = y_fwd_val.reindex(X_val_std.index)

    # Label distribution
    train_stats = {
        "buy":      (y_frac_train == 0).sum(),
        "sell":     (y_frac_train == 1).sum(),
        "no_trade": (y_frac_train == 2).sum(),
    }
    val_stats = {
        "buy":      (y_frac_val == 0).sum(),
        "sell":     (y_frac_val == 1).sum(),
        "no_trade": (y_frac_val == 2).sum(),
    }
    total_tr = len(y_frac_train)
    total_vl = len(y_frac_val)
    print(f"  Train fractal labels: "
          f"buy={train_stats['buy']}  sell={train_stats['sell']}  "
          f"no_trade={train_stats['no_trade']}  "
          f"trade_rate={100*(train_stats['buy']+train_stats['sell'])/max(total_tr,1):.1f}%")
    print(f"  Val   fractal labels: "
          f"buy={val_stats['buy']}  sell={val_stats['sell']}  "
          f"no_trade={val_stats['no_trade']}  "
          f"trade_rate={100*(val_stats['buy']+val_stats['sell'])/max(total_vl,1):.1f}%")

    # ── 4. Train standard forward-return model (baseline) ─────────────────────
    _print_sep("BASELINE: Forward-Return Labels")

    # CatBoost wants labels {0, 1, 2} — remap {-1, 0, 1} → {1, 2, 0}
    def _remap_std(y: pd.Series) -> pd.Series:
        m = {1: 0, -1: 1, 0: 2}   # buy=0, sell=1, hold=2
        return y.map(m).fillna(2).astype(int)

    y_std_train = _remap_std(y_fwd_train)
    y_std_val   = _remap_std(y_fwd_val)

    base_model = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        loss_function="MultiClass", eval_metric="Accuracy",
        verbose=0, random_seed=42,
    )
    base_model.fit(X_train_std, y_std_train)
    base_pred = base_model.predict(X_val_std).flatten()

    # Only evaluate on non-hold predictions
    mask_trade_std = y_std_val != 2
    acc_std_all  = _accuracy(y_std_val, base_pred)
    acc_std_trade = (
        _accuracy(y_std_val[mask_trade_std], base_pred[mask_trade_std])
        if mask_trade_std.sum() > 0 else float("nan")
    )
    print(f"  Val accuracy (all):      {acc_std_all:.4f}")
    print(f"  Val accuracy (trade only): {acc_std_trade:.4f}  "
          f"(n={mask_trade_std.sum()})")

    # ── 5. Train fractal direction model ──────────────────────────────────────
    _print_sep("FRACTAL: Direction Model (buy=0 / sell=1)")

    # Direction model: only train on bars where fractal says trade (0 or 1)
    trade_mask_tr  = y_frac_train != 2
    trade_mask_val = y_frac_val   != 2

    if trade_mask_tr.sum() < 20:
        print("  WARNING: too few fractal trade bars for direction model "
              f"({trade_mask_tr.sum()}). Try lower corr_threshold.")
        dir_model = None
    else:
        X_dir  = X_train_std[trade_mask_tr]
        y_dir  = y_frac_train[trade_mask_tr]  # 0=buy, 1=sell

        dir_model = CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            loss_function="Logloss", eval_metric="Accuracy",
            verbose=0, random_seed=42,
        )
        dir_model.fit(X_dir, y_dir)

        if trade_mask_val.sum() > 0:
            dir_pred = dir_model.predict(X_val_std[trade_mask_val]).flatten()
            dir_acc  = _accuracy(y_frac_val[trade_mask_val], dir_pred)
            print(f"  Val accuracy (direction): {dir_acc:.4f}  "
                  f"(n={trade_mask_val.sum()})")
        else:
            print("  No fractal trade bars in validation set.")

    # ── 6. Train fractal meta model (trade vs skip) ────────────────────────────
    _print_sep("FRACTAL: Meta Model (trade=1 / skip=0)")

    y_meta_train = (y_frac_train != 2).astype(int)
    y_meta_val   = (y_frac_val   != 2).astype(int)

    meta_model = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.05,
        loss_function="Logloss", eval_metric="F1",
        verbose=0, random_seed=42,
    )
    meta_model.fit(X_train_std, y_meta_train)
    meta_pred = meta_model.predict(X_val_std).flatten()

    meta_f1  = _f1_binary(y_meta_val.values, meta_pred, pos_label=1)
    meta_acc = _accuracy(y_meta_val.values, meta_pred)
    print(f"  Val accuracy: {meta_acc:.4f}  F1(trade): {meta_f1:.4f}")
    print(f"  Predicted trade/skip: {(meta_pred==1).sum()} / {(meta_pred==0).sum()}")

    # ── 7. Combined fractal pipeline: meta gates direction ────────────────────
    _print_sep("FRACTAL PIPELINE: Meta → Direction")

    if dir_model is not None:
        meta_trade_val = meta_pred == 1
        if meta_trade_val.sum() > 0:
            X_gated   = X_val_std[meta_trade_val]
            dir_gated = dir_model.predict(X_gated).flatten()
            y_gated   = y_frac_val[meta_trade_val]

            # Among bars predicted as "trade", how accurate is direction?
            # Only count bars where fractal label is also 0 or 1
            both_trade = meta_trade_val & (y_frac_val != 2).values
            if both_trade.sum() > 0:
                dir_on_both = dir_model.predict(X_val_std[both_trade]).flatten()
                acc_combined = _accuracy(y_frac_val[both_trade].values, dir_on_both)
                print(f"  Meta says trade AND fractal says trade: n={both_trade.sum()}")
                print(f"  Direction accuracy on those bars: {acc_combined:.4f}")
            print(f"  Total meta-selected bars: {meta_trade_val.sum()}")
        else:
            print("  Meta model predicted 0 trade bars — check corr_threshold.")

    # ── 8. Summary ────────────────────────────────────────────────────────────
    _print_sep("SUMMARY")
    print(f"  Symbol:        {args.symbol}")
    print(f"  Corr threshold:{args.corr_threshold}  "
          f"Window: {args.min_window}–{args.max_window}  "
          f"Horizon: {args.horizon}")
    print(f"  Train fractal trade rate: "
          f"{100*(train_stats['buy']+train_stats['sell'])/max(total_tr,1):.1f}%")
    print(f"  Baseline val accuracy: {acc_std_all:.4f}")
    if dir_model is not None and trade_mask_val.sum() > 0:
        print(f"  Fractal direction accuracy: {dir_acc:.4f}  "
              f"Meta F1: {meta_f1:.4f}")
    print()

    # ── 9. Optional save ──────────────────────────────────────────────────────
    if args.save:
        try:
            import joblib
            out_dir = Path(f"data/models/fractal_{args.symbol}")
            out_dir.mkdir(parents=True, exist_ok=True)
            if dir_model is not None:
                dir_model.save_model(str(out_dir / "direction_model.cbm"))
            meta_model.save_model(str(out_dir / "meta_model.cbm"))
            joblib.dump(fp, str(out_dir / "feature_pipeline.joblib"))
            import json
            cfg = {
                "symbol":         args.symbol,
                "min_window":     args.min_window,
                "max_window":     args.max_window,
                "corr_threshold": args.corr_threshold,
                "horizon":        args.horizon,
                "markup":         args.markup,
            }
            (out_dir / "fractal_config.json").write_text(json.dumps(cfg, indent=2))
            print(f"  Models saved to {out_dir}/")
        except Exception as exc:
            print(f"  Save failed: {exc}")


if __name__ == "__main__":
    main()
