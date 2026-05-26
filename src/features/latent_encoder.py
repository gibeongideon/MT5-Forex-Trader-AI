"""
Latent Feature Encoder — decoupled autoencoder for market state compression.

Sits between raw OHLCV data and the ML training pipeline. Learns compressed
"market state" representations (latent vectors) that capture trend regime,
volatility, breakout behavior, and hidden cycles — things hand-crafted
indicators miss.

Usage pattern (decoupled from all model code):
    encoder = LatentEncoder(window_size=50, latent_dim=16)
    encoder.fit(train_df)           # unsupervised, train split only
    latent_df = encoder.transform(full_df)  # adds latent_0..N columns
    encoder.save("data/models/autoencoder.pt")

No-lookahead guarantee:
    Window for bar t = ohlcv rows [t-W, t-1].
    First W-1 rows get zero latent vectors (no warm-up data available).

Within-window normalisation:
    Each window is z-scored by its own mean/std before encoding.
    This prevents price-level / scale bias across training epochs.

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
    from torch.utils.data import DataLoader, TensorDataset
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
# Internal autoencoder network
# ---------------------------------------------------------------------------

class _AutoEncoderNet(nn.Module if _TORCH_AVAILABLE else object):
    """MLP autoencoder: compress a flattened OHLCV window → latent vector."""

    def __init__(self, input_dim: int, latent_dim: int):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x: "torch.Tensor"):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

    def encode(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class LatentEncoder:
    """
    Decoupled autoencoder-based feature extractor.

    Trains on raw OHLCV sequences and transforms them into latent vectors
    that are appended to the feature matrix. Zero changes to any model code —
    models just see more columns.
    """

    OHLCV_COLS = ["open", "high", "low", "close", "tick_volume"]

    def __init__(
        self,
        window_size: int = 50,
        latent_dim: int = 16,
        epochs: int = 30,
        batch_size: int = 512,
        lr: float = 1e-3,
        random_state: int = 42,
    ):
        self.window_size = window_size
        self.latent_dim  = latent_dim
        self.epochs      = epochs
        self.batch_size  = batch_size
        self.lr          = lr
        self.random_state = random_state

        self._net: Optional[_AutoEncoderNet] = None
        self._input_dim: int = window_size * len(self.OHLCV_COLS)
        self._trained_on: Optional[str] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only the OHLCV columns, normalising column names to lower."""
        cols_lower = {c: c.lower() for c in df.columns}
        df = df.rename(columns=cols_lower)
        available = [c for c in self.OHLCV_COLS if c in df.columns]
        if len(available) < 4:
            raise ValueError(
                f"LatentEncoder needs open/high/low/close columns. "
                f"Found: {list(df.columns)}"
            )
        # Use 4 cols (no volume) if tick_volume absent
        if "tick_volume" not in df.columns:
            available = [c for c in ["open", "high", "low", "close"] if c in df.columns]
            self._input_dim = self.window_size * 4
        return df[available].astype(np.float32)

    def _build_windows(self, ohlcv: np.ndarray) -> np.ndarray:
        """
        Build sliding windows of shape (n_bars, window_size * n_cols).
        Window for bar i = rows [i-W, i-1]. First W-1 bars are skipped.
        Each window is z-scored in place.
        """
        W = self.window_size
        n, n_cols = ohlcv.shape
        windows = []
        for i in range(W, n):
            win = ohlcv[i - W : i].copy()           # shape (W, n_cols)
            # Within-window z-score normalisation
            mu  = win.mean(axis=0, keepdims=True)
            std = win.std(axis=0, keepdims=True) + 1e-8
            win = (win - mu) / std
            windows.append(win.flatten())
        return np.array(windows, dtype=np.float32)   # (n-W, W*n_cols)

    # ------------------------------------------------------------------
    # fit / transform / save / load
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "LatentEncoder":
        """
        Train the autoencoder on OHLCV data.
        Call only on the training portion of the dataset.
        """
        _require_torch()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        ohlcv = self._extract_ohlcv(df).values
        X = self._build_windows(ohlcv)              # (n-W, W*n_cols)

        input_dim = X.shape[1]
        self._input_dim = input_dim
        net = _AutoEncoderNet(input_dim, self.latent_dim)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        tensor_X = torch.from_numpy(X)
        loader = DataLoader(
            TensorDataset(tensor_X),
            batch_size=self.batch_size,
            shuffle=True,
        )

        net.train()
        print(f"[LatentEncoder] Training autoencoder — {len(X):,} windows, "
              f"input_dim={input_dim}, latent_dim={self.latent_dim}")
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon, _ = net(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)
            avg = epoch_loss / len(X)
            if epoch % 5 == 0 or epoch == 1:
                print(f"  epoch {epoch:3d}/{self.epochs}  loss={avg:.6f}")

        self._net = net
        self._trained_on = str(df.index[0]) if hasattr(df.index, '__getitem__') else "unknown"
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract latent features for every bar in df.

        Returns a DataFrame with columns latent_0 .. latent_{D-1}, aligned
        to df.index. The first window_size-1 rows are filled with zeros
        (no historical context available for those bars).
        """
        _require_torch()
        if self._net is None:
            raise RuntimeError("LatentEncoder has not been fitted. Call .fit() first.")

        ohlcv = self._extract_ohlcv(df).values
        X = self._build_windows(ohlcv)    # (n-W, W*n_cols)

        self._net.eval()
        with torch.no_grad():
            tensor_X = torch.from_numpy(X)
            z = self._net.encode(tensor_X).numpy()  # (n-W, latent_dim)

        # Pad first W-1 rows with zeros
        W = self.window_size
        pad = np.zeros((W - 1, self.latent_dim), dtype=np.float32)
        latent_values = np.vstack([pad, z])          # (n, latent_dim)

        cols = [f"latent_{i}" for i in range(self.latent_dim)]
        return pd.DataFrame(latent_values, index=df.index, columns=cols)

    def save(self, path: str) -> None:
        """Save encoder weights + config to a .pt file."""
        _require_torch()
        if self._net is None:
            raise RuntimeError("Nothing to save — call .fit() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._net.state_dict(),
                "params": {
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
        print(f"[LatentEncoder] Saved → {path}")

    def load(self, path: str) -> "LatentEncoder":
        """Load encoder weights + config from a .pt file."""
        _require_torch()
        checkpoint = torch.load(path, map_location="cpu")
        p = checkpoint["params"]
        self.window_size   = p["window_size"]
        self.latent_dim    = p["latent_dim"]
        self._input_dim    = p["input_dim"]
        self.epochs        = p["epochs"]
        self.batch_size    = p["batch_size"]
        self.lr            = p["lr"]
        self.random_state  = p["random_state"]
        self._trained_on   = checkpoint.get("trained_on")

        self._net = _AutoEncoderNet(p["input_dim"], p["latent_dim"])
        self._net.load_state_dict(checkpoint["state_dict"])
        self._net.eval()
        print(f"[LatentEncoder] Loaded ← {path}  (latent_dim={self.latent_dim})")
        return self

    def metadata(self) -> dict:
        return {
            "window_size": self.window_size,
            "latent_dim":  self.latent_dim,
            "input_dim":   self._input_dim,
            "trained_on":  self._trained_on,
            "fitted":      self._net is not None,
        }
