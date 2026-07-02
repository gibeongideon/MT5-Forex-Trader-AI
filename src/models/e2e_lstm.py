"""
End-to-end LSTM model — Phase 23-C.

Trained directly on raw 5-column OHLCV windows. No feature engineering —
the LSTM learns its own temporal representations from price/volume sequences.

Key design choices:
  - Per-window z-score normalization (within each 50-bar window, per feature).
    Absolute prices are non-stationary; within-window z-scores are stationary
    and let the model focus on relative movements rather than price levels.
  - window_size=50 matches enc8's receptive field for a fair comparison.
  - Uses the same ModelInterface contract as all other models, so it plugs
    into WalkForwardValidator and the backtester without changes.
  - Accepts any DataFrame with OHLCV columns — the model selects them
    internally. This allows the walk-forward to pass df_raw directly.

Architecture:
    Input:  (batch, 50, 5)    — 50-bar OHLCV window, per-window z-scored
    LSTM:   2-layer, hidden=64, batch_first=True, dropout=0.3
    FC:     Linear(64, 3)
    Output: softmax → [P_buy, P_hold, P_sell]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.models.model_interface import ModelInterface

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for E2ELSTMModel. "
            "Install with: conda install -n envmt5 pytorch cpuonly -c pytorch"
        )


class _E2ENet(nn.Module if _TORCH_AVAILABLE else object):
    """2-layer LSTM → FC → 3-class logits."""

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = n_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])  # last timestep: (batch, hidden)
        return self.fc(out)                    # logits: (batch, 3)


class E2ELSTMModel(ModelInterface):
    """
    End-to-end LSTM trained on raw 5-column OHLCV windows.

    Parameters
    ----------
    window_size  : bars per input window (default 50 — matches enc8)
    hidden_size  : LSTM units per layer
    num_layers   : stacked LSTM layers
    dropout      : dropout between LSTM layers and before FC
    epochs       : training passes
    batch_size   : mini-batch size
    lr           : Adam learning rate
    """

    OHLCV_COLS = ["open", "high", "low", "close", "tick_volume"]

    def __init__(
        self,
        window_size: int   = 50,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
        epochs:      int   = 30,
        batch_size:  int   = 1024,
        lr:          float = 1e-3,
        random_state: int  = 42,
    ):
        _require_torch()
        self.window_size  = window_size
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.dropout      = dropout
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.random_state = random_state

        self._net: Optional[_E2ENet] = None
        self._input_cols: list[str]  = []
        self._trained_on: str        = ""
        self._device = torch.device("cpu")

    # ── ModelInterface contract ────────────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: pd.Series) -> "E2ELSTMModel":
        """
        Train on raw OHLCV columns. X may be a raw DataFrame (with OHLCV cols)
        or a feature matrix — the model extracts whichever OHLCV cols are present.

        Labels y: -1=sell, 0=hold, 1=buy (from FeaturePipeline._make_labels convention)
        """
        torch.manual_seed(self.random_state)

        # Extract whichever OHLCV columns exist in X
        cols = [c for c in self.OHLCV_COLS if c in X.columns]
        if not cols:
            raise ValueError(
                f"E2ELSTMModel requires at least one of {self.OHLCV_COLS} in the DataFrame. "
                f"Got columns: {list(X.columns[:10])}"
            )
        self._input_cols = cols
        data = X[cols].values.astype(np.float32)  # (n_bars, n_features)

        # Align y to X
        y_aligned = y.reindex(X.index).fillna(0).astype(int)

        # Build windows: one per bar (bar i gets window [i-W, i))
        windows, labels = self._make_windows(data, y_aligned.values)

        if len(windows) == 0:
            raise ValueError("No training windows — dataset too small for window_size.")

        n_feat = windows.shape[2]
        self._net = _E2ENet(n_feat, self.hidden_size, self.num_layers, self.dropout)
        self._net.to(self._device)

        # Remap labels: -1→0 (sell), 0→1 (hold), 1→2 (buy)
        labels_mapped = np.where(labels == -1, 0, np.where(labels == 0, 1, 2))

        X_t = torch.tensor(windows, dtype=torch.float32, device=self._device)
        y_t = torch.tensor(labels_mapped, dtype=torch.long, device=self._device)
        ds  = TensorDataset(X_t, y_t)
        dl  = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self._net.train()
        for epoch in range(1, self.epochs + 1):
            total_loss, correct, total = 0.0, 0, 0
            for xb, yb in dl:
                optimizer.zero_grad()
                logits = self._net(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(yb)
                correct    += (logits.argmax(1) == yb).sum().item()
                total      += len(yb)
            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"  epoch {epoch:3d}/{self.epochs}  "
                    f"ce_loss={total_loss/total:.4f}  "
                    f"train_acc={correct/total:.1%}",
                    flush=True,
                )

        self._trained_on = f"{X.index[0].date()} → {X.index[-1].date()}"
        self._net.eval()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict [P_buy, P_hold, P_sell] for every bar in X.

        Returns
        -------
        Array shape (n_rows, 3) with per-bar probabilities.
        Rows with insufficient history (< window_size bars) get uniform 1/3.
        """
        if self._net is None:
            raise RuntimeError("E2ELSTMModel not trained. Call train() first.")

        cols = self._input_cols or [c for c in self.OHLCV_COLS if c in X.columns]
        data = X[cols].values.astype(np.float32)
        n    = len(data)
        W    = self.window_size

        out  = np.full((n, 3), 1.0 / 3.0, dtype=np.float32)

        if n < W:
            return out

        # Build windows for all bars that have enough history
        windows, indices = [], []
        for i in range(W, n + 1):
            win = data[i - W: i]          # (W, n_feat)
            win = self._normalize_window(win)
            windows.append(win)
            indices.append(i - 1)         # prediction for bar at index i-1

        X_t = torch.tensor(np.array(windows), dtype=torch.float32, device=self._device)

        self._net.eval()
        with torch.no_grad():
            logits = self._net(X_t)          # (n_windows, 3)
            proba  = torch.softmax(logits, dim=1).cpu().numpy()

        # proba columns: [P_sell, P_hold, P_buy] (from label mapping 0→sell, 1→hold, 2→buy)
        # reorder to ModelInterface convention: [P_buy, P_hold, P_sell]
        proba_reordered = proba[:, [2, 1, 0]]

        for k, idx in enumerate(indices):
            out[idx] = proba_reordered[k]

        return out

    def save(self, path: str | Path) -> None:
        _require_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict":  self._net.state_dict(),
            "input_cols":  self._input_cols,
            "trained_on":  self._trained_on,
            "params": {
                "window_size": self.window_size,
                "hidden_size": self.hidden_size,
                "num_layers":  self.num_layers,
                "dropout":     self.dropout,
                "epochs":      self.epochs,
                "batch_size":  self.batch_size,
                "lr":          self.lr,
            },
        }, path)
        print(f"E2ELSTMModel saved → {path}")

    def load(self, path: str | Path) -> None:
        _require_torch()
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        p    = ckpt.get("params", {})
        self.window_size = p.get("window_size", self.window_size)
        self.hidden_size = p.get("hidden_size", self.hidden_size)
        self.num_layers  = p.get("num_layers",  self.num_layers)
        self.dropout     = p.get("dropout",     self.dropout)
        self._input_cols = ckpt.get("input_cols", self.OHLCV_COLS)
        self._trained_on = ckpt.get("trained_on", "")

        n_feat = len(self._input_cols)
        self._net = _E2ENet(n_feat, self.hidden_size, self.num_layers, self.dropout)
        self._net.load_state_dict(ckpt["state_dict"])
        self._net.to(self._device)
        self._net.eval()
        print(f"E2ELSTMModel loaded ← {path}")

    def metadata(self) -> dict:
        return {
            "type":        "e2e_lstm",
            "window_size": self.window_size,
            "hidden_size": self.hidden_size,
            "num_layers":  self.num_layers,
            "input_cols":  self._input_cols,
            "trained_on":  self._trained_on,
            "fitted":      self._net is not None,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_windows(
        self,
        data:   np.ndarray,    # (n_bars, n_features) float32
        labels: np.ndarray,    # (n_bars,) int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build per-window-normalized windows and aligned labels."""
        W   = self.window_size
        n   = len(data)
        wins, labs = [], []
        for i in range(W, n + 1):
            win = data[i - W: i].copy()       # (W, n_feat)
            win = self._normalize_window(win)
            wins.append(win)
            labs.append(labels[i - 1])         # label for the last bar of this window
        if not wins:
            return np.empty((0, W, data.shape[1])), np.empty(0, dtype=int)
        return np.array(wins, dtype=np.float32), np.array(labs, dtype=int)

    @staticmethod
    def _normalize_window(win: np.ndarray) -> np.ndarray:
        """Per-window, per-feature z-score normalization."""
        mu  = win.mean(axis=0, keepdims=True)
        std = win.std(axis=0, keepdims=True) + 1e-8
        return (win - mu) / std
