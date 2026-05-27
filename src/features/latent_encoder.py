"""
Latent Feature Encoder — two training modes.

MODE 1: autoencoder (unsupervised)
    Encoder learns to reconstruct raw OHLCV windows.
    Latent features capture market structure, but are NOT guaranteed
    to correlate with future price direction.

MODE 2: supervised (default, recommended)
    Encoder + classification head trained jointly with CrossEntropyLoss
    on the buy/hold/sell labels. Latent features are forced to encode
    information that is predictive of next-bar direction, not just
    reconstruction quality.

Usage:
    # Supervised (default — trains on direction labels)
    enc = LatentEncoder(mode="supervised", latent_dim=8)
    enc.fit(train_df, y_train)    # y_train: pd.Series of -1/0/1 aligned to df
    latent_df = enc.transform(full_df)

    # Autoencoder (unsupervised, no labels needed)
    enc = LatentEncoder(mode="autoencoder", latent_dim=16)
    enc.fit(train_df)
    latent_df = enc.transform(full_df)

No-lookahead guarantee:
    Window for bar t = ohlcv rows [t-W, t-1].
    First W rows get zero latent vectors.

Requires: torch (conda install -n envmt5 pytorch cpuonly -c pytorch)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for LatentEncoder. "
            "Install with: conda install -n envmt5 pytorch cpuonly -c pytorch"
        )


# ---------------------------------------------------------------------------
# Internal networks
# ---------------------------------------------------------------------------

class _AutoEncoderNet(nn.Module if _TORCH_AVAILABLE else object):
    """MLP autoencoder: window → latent → reconstruct window. MSE loss."""

    def __init__(self, input_dim: int, latent_dim: int):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256),        nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x: "torch.Tensor"):
        z   = self.encoder(x)
        out = self.decoder(z)
        return out, z

    def encode(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.encoder(x)


class _SupervisedEncoderNet(nn.Module if _TORCH_AVAILABLE else object):
    """Encoder + classification head trained end-to-end on direction labels.

    After training the head is discarded; only encoder weights are used
    during transform(). This forces the latent space to encode features
    that are predictive of buy/hold/sell, not just reconstructive.
    """

    def __init__(self, input_dim: int, latent_dim: int, n_classes: int = 3):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, latent_dim), nn.Tanh(),  # Tanh bounds latent space
        )
        self.head = nn.Linear(latent_dim, n_classes)  # logit head, no softmax

    def forward(self, x: "torch.Tensor"):
        z      = self.encoder(x)
        logits = self.head(z)
        return logits, z

    def encode(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class LatentEncoder:
    """
    Decoupled latent feature extractor.

    Parameters
    ----------
    mode        : "supervised" (default) or "autoencoder"
    window_size : OHLCV bars per input window (bars t-W … t-1)
    latent_dim  : size of the latent vector (8 recommended for supervised)
    epochs      : training epochs
    batch_size  : mini-batch size
    lr          : Adam learning rate
    random_state: seed
    """

    OHLCV_COLS = ["open", "high", "low", "close", "tick_volume"]
    # Label remap: {-1: sell, 0: hold, 1: buy} → {0, 1, 2} for CrossEntropy
    _LABEL_MAP = {-1: 0, 0: 1, 1: 2}

    def __init__(
        self,
        mode:         str   = "supervised",
        window_size:  int   = 50,
        latent_dim:   int   = 8,
        epochs:       int   = 30,
        batch_size:   int   = 4096,
        lr:           float = 1e-3,
        random_state: int   = 42,
    ):
        if mode not in ("supervised", "autoencoder"):
            raise ValueError('mode must be "supervised" or "autoencoder"')
        self.mode         = mode
        self.window_size  = window_size
        self.latent_dim   = latent_dim
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.random_state = random_state

        self._net: Optional[object] = None
        self._input_dim: int = window_size * len(self.OHLCV_COLS)
        self._trained_on: Optional[str] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_lower = {c: c.lower() for c in df.columns}
        df = df.rename(columns=cols_lower)
        available = [c for c in self.OHLCV_COLS if c in df.columns]
        if len(available) < 4:
            raise ValueError(
                f"LatentEncoder needs open/high/low/close. Found: {list(df.columns)}"
            )
        if "tick_volume" not in df.columns:
            available = [c for c in ["open", "high", "low", "close"] if c in df.columns]
            self._input_dim = self.window_size * 4
        return df[available].astype(np.float32)

    def _build_windows(self, ohlcv: np.ndarray) -> np.ndarray:
        """Vectorised sliding windows via stride_tricks. No Python loop."""
        W = self.window_size
        n, n_cols = ohlcv.shape
        shape   = (n - W, W, n_cols)
        strides = (ohlcv.strides[0], ohlcv.strides[0], ohlcv.strides[1])
        windows = np.lib.stride_tricks.as_strided(
            ohlcv, shape=shape, strides=strides
        ).copy().astype(np.float32)
        # Within-window z-score (axis=1 = time axis)
        mu  = windows.mean(axis=1, keepdims=True)
        std = windows.std(axis=1, keepdims=True) + 1e-8
        windows = (windows - mu) / std
        return windows.reshape(n - W, -1)  # (n-W, W*n_cols)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> "LatentEncoder":
        """
        Train the encoder.

        Parameters
        ----------
        df : raw OHLCV DataFrame (training split only — no test data)
        y  : direction labels pd.Series of {-1, 0, 1} aligned to df.index.
             Required for mode="supervised", ignored for mode="autoencoder".
        """
        if self.mode == "supervised" and y is None:
            raise ValueError(
                'mode="supervised" requires labels y. '
                'Pass y=labels_series, or use mode="autoencoder".'
            )
        _require_torch()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        ohlcv_df = self._extract_ohlcv(df)
        ohlcv    = ohlcv_df.values
        X        = self._build_windows(ohlcv)     # (n-W, W*n_cols)
        input_dim = X.shape[1]
        self._input_dim = input_dim

        if self.mode == "supervised":
            self._fit_supervised(X, ohlcv_df.index, y)
        else:
            self._fit_autoencoder(X)

        self._trained_on = str(df.index[0]) if hasattr(df.index, '__getitem__') else "unknown"
        return self

    def _fit_supervised(
        self,
        X: np.ndarray,
        ohlcv_index: "pd.Index",
        y: pd.Series,
    ) -> None:
        """Train encoder + classification head jointly on direction labels."""
        W = self.window_size
        # Bar times for each window: window j → bar at ohlcv_index[W + j]
        bar_times = ohlcv_index[W:]              # length = n - W = len(X)
        labels    = y.reindex(bar_times)         # align to window times
        valid     = labels.notna()
        X_valid   = X[valid.values]
        y_valid   = labels[valid].values.astype(int)
        y_mapped  = np.array(
            [self._LABEL_MAP[int(l)] for l in y_valid], dtype=np.int64
        )

        net       = _SupervisedEncoderNet(self._input_dim, self.latent_dim)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        tensor_X = torch.from_numpy(X_valid)
        tensor_y = torch.from_numpy(y_mapped)
        n        = len(tensor_X)

        # Class weights: inverse-frequency to handle imbalance
        counts = np.bincount(y_mapped, minlength=3).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        weights = torch.tensor(1.0 / counts / (1.0 / counts).sum(), dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weights)

        net.train()
        print(
            f"[LatentEncoder] Supervised training — {n:,} labelled windows, "
            f"input_dim={self._input_dim}, latent_dim={self.latent_dim}",
            flush=True,
        )
        label_dist = {k: int((y_mapped == v).sum()) for k, v in [('sell',0),('hold',1),('buy',2)]}
        print(f"  Label distribution: {label_dist}", flush=True)

        for epoch in range(1, self.epochs + 1):
            perm        = torch.randperm(n)
            epoch_loss  = 0.0
            correct     = 0
            for start in range(0, n, self.batch_size):
                idx    = perm[start : start + self.batch_size]
                bx, by = tensor_X[idx], tensor_y[idx]
                optimizer.zero_grad()
                logits, _ = net(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(bx)
                correct    += (logits.argmax(1) == by).sum().item()
            avg_loss = epoch_loss / n
            acc      = correct / n
            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"  epoch {epoch:3d}/{self.epochs}  "
                    f"ce_loss={avg_loss:.4f}  train_acc={acc:.1%}",
                    flush=True,
                )

        self._net = net

    def _fit_autoencoder(self, X: np.ndarray) -> None:
        """Train autoencoder with MSE reconstruction loss."""
        net       = _AutoEncoderNet(self._input_dim, self.latent_dim)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        tensor_X  = torch.from_numpy(X)
        n         = len(tensor_X)

        net.train()
        print(
            f"[LatentEncoder] Autoencoder training — {n:,} windows, "
            f"input_dim={self._input_dim}, latent_dim={self.latent_dim}",
            flush=True,
        )

        for epoch in range(1, self.epochs + 1):
            perm       = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                idx   = perm[start : start + self.batch_size]
                batch = tensor_X[idx]
                optimizer.zero_grad()
                recon, _ = net(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)
            avg = epoch_loss / n
            if epoch % 5 == 0 or epoch == 1:
                print(f"  epoch {epoch:3d}/{self.epochs}  mse_loss={avg:.6f}", flush=True)

        self._net = net

    # ------------------------------------------------------------------
    # transform
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract latent features for every bar in df.
        Returns DataFrame with columns latent_0 .. latent_{D-1}.
        First window_size rows are filled with zeros (no lookback available).
        """
        _require_torch()
        if self._net is None:
            raise RuntimeError("Not fitted. Call .fit() first.")

        ohlcv = self._extract_ohlcv(df).values
        X     = self._build_windows(ohlcv)

        self._net.eval()
        with torch.no_grad():
            z = self._net.encode(torch.from_numpy(X)).numpy()  # (n-W, latent_dim)

        W   = self.window_size
        pad = np.zeros((W, self.latent_dim), dtype=np.float32)
        latent_values = np.vstack([pad, z])   # (n, latent_dim)

        cols = [f"latent_{i}" for i in range(self.latent_dim)]
        return pd.DataFrame(latent_values, index=df.index, columns=cols)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        _require_torch()
        if self._net is None:
            raise RuntimeError("Nothing to save — call .fit() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "params": {
                    "mode":         self.mode,
                    "window_size":  self.window_size,
                    "latent_dim":   self.latent_dim,
                    "input_dim":    self._input_dim,
                    "epochs":       self.epochs,
                    "batch_size":   self.batch_size,
                    "lr":           self.lr,
                    "random_state": self.random_state,
                },
                "trained_on": self._trained_on,
            },
            path,
        )
        print(f"[LatentEncoder] Saved → {path}", flush=True)

    def load(self, path: str) -> "LatentEncoder":
        _require_torch()
        ckpt = torch.load(path, map_location="cpu")
        p    = ckpt["params"]

        self.mode         = p.get("mode", "autoencoder")
        self.window_size  = p["window_size"]
        self.latent_dim   = p["latent_dim"]
        self._input_dim   = p["input_dim"]
        self.epochs       = p["epochs"]
        self.batch_size   = p["batch_size"]
        self.lr           = p["lr"]
        self.random_state = p["random_state"]
        self._trained_on  = ckpt.get("trained_on")

        if self.mode == "supervised":
            net = _SupervisedEncoderNet(p["input_dim"], p["latent_dim"])
        else:
            net = _AutoEncoderNet(p["input_dim"], p["latent_dim"])

        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        self._net = net
        print(
            f"[LatentEncoder] Loaded ← {path}  "
            f"(mode={self.mode}, latent_dim={self.latent_dim})",
            flush=True,
        )
        return self

    def metadata(self) -> dict:
        return {
            "mode":        self.mode,
            "window_size": self.window_size,
            "latent_dim":  self.latent_dim,
            "input_dim":   self._input_dim,
            "trained_on":  self._trained_on,
            "fitted":      self._net is not None,
        }
