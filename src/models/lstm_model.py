"""
LSTM prediction model — Phase 6.

A 2-layer LSTM that processes sequences of bars and outputs calibrated
[P_buy, P_hold, P_sell] probabilities. Unlike the tree-based models which
treat each bar independently, LSTM has memory — bar[t] sees the context
of the previous `seq_len` bars.

This gives the ensemble a signal that is structurally different from
XGBoost/LightGBM/CatBoost, improving stacking diversity.

Requires: torch (pip/conda install pytorch)

Label convention (must match feature_pipeline.py):
    y = 1  → buy
    y = 0  → hold
    y = -1 → sell

Output convention (ModelInterface contract):
    predict_proba() → [P_buy, P_hold, P_sell]
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
            "PyTorch is required for LSTMModel. "
            "Install with: conda install -n envmt5 pytorch cpuonly -c pytorch"
        )


class _LSTMNet(nn.Module if _TORCH_AVAILABLE else object):
    """2-layer LSTM → fully-connected → 3-class softmax."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])   # take last timestep
        return self.fc(out)                     # logits, shape (batch, 3)


class LSTMModel(ModelInterface):
    """
    2-layer LSTM 3-class classifier.

    Parameters
    ----------
    seq_len      : number of bars fed as one sequence (lookback window)
    hidden_size  : LSTM hidden units per layer
    num_layers   : number of LSTM layers stacked
    dropout      : dropout rate between LSTM layers
    epochs       : training epochs
    batch_size   : mini-batch size
    lr           : Adam learning rate
    """

    def __init__(
        self,
        seq_len:     int   = 20,
        hidden_size: int   = 64,
        num_layers:  int   = 2,
        dropout:     float = 0.2,
        epochs:      int   = 30,
        batch_size:  int   = 256,
        lr:          float = 1e-3,
        random_state: int  = 42,
    ):
        self.seq_len     = seq_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.random_state = random_state

        self._net: Optional[object] = None
        self._feature_names: list[str] = []
        self._classes: np.ndarray | None = None
        self._trained_on: str = ""
        self._n_features: int = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_sequences(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ):
        """Slide a window of seq_len over rows → (n_seq, seq_len, n_feat)."""
        n = len(X)
        xs, ys = [], []
        for i in range(self.seq_len, n):
            xs.append(X[i - self.seq_len : i])
            if y is not None:
                ys.append(y[i])
        X_seq = np.array(xs, dtype=np.float32)
        if y is not None:
            return X_seq, np.array(ys, dtype=np.int64)
        return X_seq

    def _label_to_idx(self, y: np.ndarray) -> np.ndarray:
        """Remap -1/0/1 → 0/1/2 for cross-entropy loss."""
        return np.searchsorted(self._classes, y)

    # ── ModelInterface ────────────────────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: pd.Series) -> "LSTMModel":
        _require_torch()
        torch.manual_seed(self.random_state)

        self._feature_names = list(X.columns)
        self._trained_on    = f"{X.index[0].date()} → {X.index[-1].date()}"
        self._classes       = np.sort(np.unique(y.values))
        self._n_features    = X.shape[1]

        X_np = X.values.astype(np.float32)
        y_np = self._label_to_idx(y.values)

        X_seq, y_seq = self._make_sequences(X_np, y_np)

        dataset = TensorDataset(
            torch.from_numpy(X_seq),
            torch.from_numpy(y_seq),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        net = _LSTMNet(self._n_features, self.hidden_size, self.num_layers, self.dropout)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        net.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = net(xb)
                loss   = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                avg = total_loss / len(loader)
                print(f"  LSTM epoch {epoch+1}/{self.epochs}  loss={avg:.4f}")

        self._net = net
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns shape (n_rows, 3) — columns: [P_buy, P_hold, P_sell].
        For a single-row input, returns shape (3,).
        Note: first seq_len rows cannot be predicted (no history) → returns
        uniform [1/3, 1/3, 1/3] for those rows.
        """
        _require_torch()
        if self._net is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        cols = [c for c in self._feature_names if c in X.columns]
        X_np = X[cols].values.astype(np.float32)
        n    = len(X_np)

        # Rows we can predict
        X_seq = self._make_sequences(X_np)   # shape (n - seq_len, seq_len, feat)

        self._net.eval()
        with torch.no_grad():
            logits = self._net(torch.from_numpy(X_seq))
            probs_raw = torch.softmax(logits, dim=1).numpy()  # (n-seq_len, 3)

        # Reorder columns to [P_buy, P_hold, P_sell]
        class_list = list(self._classes)
        idx_buy  = class_list.index(1)  if 1  in class_list else None
        idx_hold = class_list.index(0)  if 0  in class_list else None
        idx_sell = class_list.index(-1) if -1 in class_list else None

        ordered = np.full((len(probs_raw), 3), 1/3, dtype=np.float32)
        if idx_buy  is not None: ordered[:, 0] = probs_raw[:, idx_buy]
        if idx_hold is not None: ordered[:, 1] = probs_raw[:, idx_hold]
        if idx_sell is not None: ordered[:, 2] = probs_raw[:, idx_sell]

        # Pad the first seq_len rows with uniform probabilities
        pad    = np.full((self.seq_len, 3), 1/3, dtype=np.float32)
        result = np.vstack([pad, ordered])[:n]

        return result[0] if n == 1 else result

    def save(self, path: str | Path = "data/models/lstm.pt") -> None:
        _require_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict":    self._net.state_dict() if self._net else None,
            "feature_names": self._feature_names,
            "classes":       self._classes,
            "trained_on":    self._trained_on,
            "n_features":    self._n_features,
            "params": {
                "seq_len":     self.seq_len,
                "hidden_size": self.hidden_size,
                "num_layers":  self.num_layers,
                "dropout":     self.dropout,
                "epochs":      self.epochs,
                "batch_size":  self.batch_size,
                "lr":          self.lr,
            },
        }, path)
        print(f"Model saved → {path}")

    def load(self, path: str | Path = "data/models/lstm.pt") -> "LSTMModel":
        _require_torch()
        checkpoint = torch.load(path, map_location="cpu")
        params = checkpoint.get("params", {})
        for k, v in params.items():
            setattr(self, k, v)
        self._feature_names = checkpoint["feature_names"]
        self._classes       = checkpoint["classes"]
        self._trained_on    = checkpoint.get("trained_on", "")
        self._n_features    = checkpoint["n_features"]
        net = _LSTMNet(self._n_features, self.hidden_size, self.num_layers, self.dropout)
        if checkpoint["state_dict"] is not None:
            net.load_state_dict(checkpoint["state_dict"])
        self._net = net
        print(f"Model loaded ← {path}")
        return self

    def metadata(self) -> dict:
        return {
            "name":       "LSTMModel",
            "version":    "1.0",
            "trained_on": self._trained_on,
            "features":   self._feature_names,
            "n_classes":  3,
            "params": {
                "seq_len":     self.seq_len,
                "hidden_size": self.hidden_size,
                "num_layers":  self.num_layers,
                "epochs":      self.epochs,
            },
        }
