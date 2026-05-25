"""
Bar Language Model — Phase 9 (local, optional).

A small Transformer encoder (~200k params) trained on discretised OHLCV token
sequences.  Each bar is represented as one integer token (from BarTokenizer's
45-token vocabulary); the model learns sequential patterns in this "bar language".

Architecture:
    Token Embedding (47 vocab, d_model=32)
    + Learned Positional Encoding
    → 4 × TransformerEncoderLayer (4 heads, ff_dim=64)
    → Mean-pool over sequence
    → Linear(32 → 3) + Softmax
    → [P_buy, P_hold, P_sell]

~200k parameters — trainable on 50k bars without overfitting.

This model is complementary to LLMSignalModel: it runs fully offline, is fast
at inference (~1ms/bar), and captures short-range sequential structure that tree
models miss.

Requires: torch (conda install pytorch)

Usage:
    model = BarLMModel()
    model.train(X, y)
    proba = model.predict_proba(X)   # (N, 3)
    model.save("data/models/bar_lm.pt")
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.model_interface import ModelInterface
from src.features.bar_tokenizer import BarTokenizer, VOCAB_SIZE, PAD_ID

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
            "PyTorch is required for BarLMModel. "
            "Install with: conda install -n envmt5 pytorch cpuonly -c pytorch"
        )


# ── Transformer model ──────────────────────────────────────────────────────────

class _BarTransformer(nn.Module if _TORCH_AVAILABLE else object):
    """
    Tiny Transformer encoder for token sequences.
    Input : (batch, seq_len) integer token ids
    Output: (batch, 3) logits
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:    int,
        n_heads:    int,
        n_layers:   int,
        ff_dim:     int,
        max_len:    int,
        dropout:    float,
        pad_id:     int,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        super().__init__()
        self.pad_id = pad_id

        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos   = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model      = d_model,
            nhead        = n_heads,
            dim_feedforward = ff_dim,
            dropout      = dropout,
            batch_first  = True,
            norm_first   = True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head    = nn.Linear(d_model, 3)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        # x: (batch, seq_len) integer ids
        batch, seq = x.shape
        positions  = torch.arange(seq, device=x.device).unsqueeze(0).expand(batch, -1)
        padding_mask = (x == self.pad_id)  # True where padded → ignored by attention

        emb = self.embed(x) + self.pos(positions)  # (batch, seq, d_model)
        out = self.encoder(emb, src_key_padding_mask=padding_mask)

        # Mean-pool over non-padding positions
        non_pad = (~padding_mask).float().unsqueeze(-1)   # (batch, seq, 1)
        pooled  = (out * non_pad).sum(1) / non_pad.sum(1).clamp(min=1e-6)
        return self.head(pooled)                           # (batch, 3)


# ── BarLMModel ─────────────────────────────────────────────────────────────────

class BarLMModel(ModelInterface):
    """
    Tiny Transformer trained on BarTokenizer integer sequences.

    Parameters
    ----------
    vocab_size  : Token vocabulary size (must match BarTokenizer.VOCAB_SIZE = 47)
    seq_len     : Lookback window in bars (tokens per sequence)
    d_model     : Embedding / hidden dimension
    n_heads     : Attention heads (d_model must be divisible by n_heads)
    n_layers    : TransformerEncoder layers
    ff_dim      : Feed-forward inner dimension
    epochs      : Training epochs
    batch_size  : Mini-batch size
    lr          : Adam learning rate
    dropout     : Dropout in encoder layers
    """

    def __init__(
        self,
        vocab_size:  int   = VOCAB_SIZE,
        seq_len:     int   = 32,
        d_model:     int   = 32,
        n_heads:     int   = 4,
        n_layers:    int   = 4,
        ff_dim:      int   = 64,
        epochs:      int   = 30,
        batch_size:  int   = 256,
        lr:          float = 1e-3,
        dropout:     float = 0.1,
        random_state: int  = 42,
    ):
        self.vocab_size   = vocab_size
        self.seq_len      = seq_len
        self.d_model      = d_model
        self.n_heads      = n_heads
        self.n_layers     = n_layers
        self.ff_dim       = ff_dim
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.dropout      = dropout
        self.random_state = random_state

        self._net:           Optional[object]   = None
        self._tokenizer:     BarTokenizer       = BarTokenizer()
        self._feature_names: list[str]          = []
        self._classes:       Optional[np.ndarray] = None
        self._trained_on:    str                = ""

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_sequences(
        self,
        ids: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        Sliding window over integer id array.
        Returns (X_seq: int64 (N, seq_len), y_seq: int64 (N,)) or just X_seq.
        """
        n = len(ids)
        xs, ys = [], []
        for i in range(self.seq_len, n):
            xs.append(ids[i - self.seq_len: i])
            if labels is not None:
                ys.append(labels[i])
        X_seq = np.array(xs, dtype=np.int64)
        if labels is not None:
            return X_seq, np.array(ys, dtype=np.int64)
        return X_seq, None

    def _label_to_idx(self, y: np.ndarray) -> np.ndarray:
        return np.searchsorted(self._classes, y)

    def _idx_to_label_order(self) -> list[int]:
        """
        Map internal class indices back to [P_buy, P_hold, P_sell] column order.
        Classes are sorted: -1 → sell(idx=0), 0 → hold(idx=1), 1 → buy(idx=2)
        Output order: [P_buy, P_hold, P_sell] = [idx2, idx1, idx0]
        """
        # classes sorted: [-1, 0, 1] → indices [0, 1, 2]
        # P_buy = class 1 = sorted index 2
        # P_hold = class 0 = sorted index 1
        # P_sell = class -1 = sorted index 0
        return [2, 1, 0]

    # ── ModelInterface ─────────────────────────────────────────────────────────

    def train(self, X: pd.DataFrame, y: pd.Series) -> "BarLMModel":
        _require_torch()
        torch.manual_seed(self.random_state)

        self._feature_names = list(X.columns)
        self._trained_on    = f"{X.index[0].date()} → {X.index[-1].date()}"
        self._classes       = np.sort(np.unique(y.values))

        # Fit tokenizer on training OHLCV from X (uses atr_14 if present)
        # X may not have open/high/low/close; tokenizer encodes via atr_14 + close proxy
        # For training we still need bar-level ids — use encode_all_ids with X's OHLCV proxy
        # Build a combined df for the tokenizer (X contains atr_14, rsi_14, etc.)
        self._tokenizer.fit(X)  # learns body size bins from return_1 × atr_14 proxy

        # Encode all bars to integer ids using available features
        ids = self._bars_to_ids(X)

        # Align labels with sequence targets (label for bar at position i)
        label_idx = self._label_to_idx(y.values)

        X_seq, y_seq = self._make_sequences(ids, label_idx)
        if len(X_seq) == 0:
            raise ValueError(f"Not enough bars ({len(ids)}) for seq_len={self.seq_len}")

        dataset = TensorDataset(
            torch.from_numpy(X_seq),
            torch.from_numpy(y_seq),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                            drop_last=False)

        net = _BarTransformer(
            vocab_size = self.vocab_size,
            d_model    = self.d_model,
            n_heads    = self.n_heads,
            n_layers   = self.n_layers,
            ff_dim     = self.ff_dim,
            max_len    = self.seq_len + 1,
            dropout    = self.dropout,
            pad_id     = PAD_ID,
        )
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)
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
                print(f"  BarLM epoch {epoch+1}/{self.epochs}  loss={avg:.4f}")

        self._net = net
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns (N, 3) [P_buy, P_hold, P_sell].
        First seq_len rows → uniform [1/3, 1/3, 1/3] (insufficient history).
        Single-row input → shape (3,).
        """
        _require_torch()
        if self._net is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        n   = len(X)
        ids = self._bars_to_ids(X)

        X_seq, _ = self._make_sequences(ids)          # (n - seq_len, seq_len)
        result   = np.full((n, 3), 1.0 / 3.0)

        if X_seq is not None and len(X_seq) > 0:
            self._net.eval()
            with torch.no_grad():
                logits   = self._net(torch.from_numpy(X_seq.astype(np.int64)))
                probs_raw = torch.softmax(logits, dim=1).numpy()  # (n-seq_len, 3)

            # Reorder to [P_buy, P_hold, P_sell]
            order = self._idx_to_label_order()
            probs_ordered = probs_raw[:, order]       # (n-seq_len, 3)

            # Write into result; first seq_len rows stay uniform
            result[self.seq_len:] = probs_ordered

        return result[0] if n == 1 else result

    def save(self, path) -> None:
        _require_torch()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "net_state":     self._net.state_dict() if self._net else None,
            "feature_names": self._feature_names,
            "classes":       self._classes,
            "trained_on":    self._trained_on,
            "tokenizer_bin_edges": self._tokenizer._bin_edges,
            "params": {
                "vocab_size":  self.vocab_size,
                "seq_len":     self.seq_len,
                "d_model":     self.d_model,
                "n_heads":     self.n_heads,
                "n_layers":    self.n_layers,
                "ff_dim":      self.ff_dim,
                "epochs":      self.epochs,
                "batch_size":  self.batch_size,
                "lr":          self.lr,
                "dropout":     self.dropout,
            },
        }, path)

    def load(self, path) -> "BarLMModel":
        _require_torch()
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        p    = ckpt["params"]

        self.vocab_size  = p["vocab_size"]
        self.seq_len     = p["seq_len"]
        self.d_model     = p["d_model"]
        self.n_heads     = p["n_heads"]
        self.n_layers    = p["n_layers"]
        self.ff_dim      = p["ff_dim"]
        self.epochs      = p["epochs"]
        self.batch_size  = p["batch_size"]
        self.lr          = p["lr"]
        self.dropout     = p.get("dropout", 0.1)

        self._feature_names = ckpt.get("feature_names", [])
        self._classes       = ckpt.get("classes")
        self._trained_on    = ckpt.get("trained_on", "")

        # Restore tokenizer bin edges
        bin_edges = ckpt.get("tokenizer_bin_edges")
        if bin_edges is not None:
            self._tokenizer._bin_edges = bin_edges
            self._tokenizer._is_fitted = True

        if ckpt.get("net_state") is not None:
            net = _BarTransformer(
                vocab_size = self.vocab_size,
                d_model    = self.d_model,
                n_heads    = self.n_heads,
                n_layers   = self.n_layers,
                ff_dim     = self.ff_dim,
                max_len    = self.seq_len + 1,
                dropout    = self.dropout,
                pad_id     = PAD_ID,
            )
            net.load_state_dict(ckpt["net_state"])
            net.eval()
            self._net = net
        return self

    def metadata(self) -> dict:
        return {
            "name":       "BarLMModel",
            "version":    "1.0",
            "trained_on": self._trained_on,
            "features":   self._feature_names,
            "n_classes":  3,
            "params": {
                "vocab_size": self.vocab_size,
                "seq_len":    self.seq_len,
                "d_model":    self.d_model,
                "n_layers":   self.n_layers,
                "epochs":     self.epochs,
            },
        }

    # ── Tokenisation helper ────────────────────────────────────────────────────

    def _bars_to_ids(self, X: pd.DataFrame) -> np.ndarray:
        """
        Convert feature matrix rows to integer token ids.
        We proxy OHLCV from feature columns:
          - close proxy: close_lag_1 shifted back (or use return_1 cumsum)
          - atr: atr_14 from feature matrix
        If X doesn't have the required columns we return UNK ids.
        """
        from src.features.bar_tokenizer import UNK_ID
        n = len(X)
        ids = np.full(n, UNK_ID, dtype=np.int64)

        has_atr    = "atr_14" in X.columns
        has_return = "return_1" in X.columns

        if not (has_atr and has_return):
            return ids

        atr_vals    = X["atr_14"].values
        ret_vals    = X["return_1"].values      # close/prev_close - 1
        # Derive |body| / ATR ≈ |return_1| * prev_close / atr — use return as proxy
        # For direction: return_1 > 0 → UP, < 0 → DN
        # For body size: |return_1| * close / atr ≈ body/atr
        # For wick: use rolling_std as proxy (high wick → higher intrabar variance)
        has_std  = "rolling_std_10" in X.columns
        has_bb   = "bb_pct" in X.columns

        for i in range(n):
            atr = float(atr_vals[i])
            ret = float(ret_vals[i])
            if atr <= 0 or np.isnan(ret):
                continue

            # Direction
            body_ratio = abs(ret) / (atr / 10_000 + 1e-8)  # normalised approximation
            if abs(ret) < 1e-5:
                direction = "DJ"
            elif ret > 0:
                direction = "UP"
            else:
                direction = "DN"

            size = self._tokenizer._body_ratio_to_size(body_ratio)

            # Wick proxy: use rolling_std asymmetry if available
            if has_std:
                std = float(X["rolling_std_10"].iloc[i])
                wick_asymm = std / (abs(ret) + 1e-6)
                if has_bb:
                    bb_pct = float(X["bb_pct"].iloc[i])
                    wick = "TW" if bb_pct > 0.8 else ("BW" if bb_pct < 0.2 else "NW")
                else:
                    wick = "TW" if wick_asymm > 2 and ret > 0 else (
                           "BW" if wick_asymm > 2 and ret < 0 else "NW")
            else:
                wick = "NW"

            tok = f"{direction}_{size}_{wick}"
            ids[i] = self._tokenizer.vocab.get(tok, 1)  # 1 = UNK

        return ids
