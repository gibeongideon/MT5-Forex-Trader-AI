"""
Latent Feature Encoder — four training modes.

MODE 1: autoencoder (unsupervised)
    Encoder learns to reconstruct raw OHLCV windows.

MODE 2: supervised (default — current best +3.13 Sharpe)
    MLP encoder + direction classification head, CrossEntropyLoss on buy/hold/sell.

MODE 3: transformer
    Self-attention encoder (drop-in for supervised). Needs 150+ epochs + LR warmup.

MODE 4: multitask (EXPERIMENT — may improve over supervised)
    Same MLP encoder as supervised, but adds a second auxiliary head that
    simultaneously predicts next-bar normalized volatility (MSE loss).
    Total loss = L_direction + alpha × L_volatility
    Forces the encoder to capture both directional AND risk-level information.
    After training, only the encoder trunk is kept — both heads are discarded.

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
    """MLP encoder + classification head trained end-to-end on direction labels.

    After training the head is discarded; only encoder weights are used
    during transform(). This forces the latent space to encode features
    that are predictive of buy/hold/sell, not just reconstructive.
    Current best: Sharpe +3.13 with latent_dim=8.
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


class _MultiTaskEncoderNet(nn.Module if _TORCH_AVAILABLE else object):
    """MLP encoder with TWO heads: direction classification + volatility regression.

    Primary   : direction head (CrossEntropyLoss) — same as _SupervisedEncoderNet
    Auxiliary : volatility head (MSELoss) — predicts normalized next-bar move size

    Training loss: L_dir + alpha × L_vol
    The shared encoder receives gradients from both, forcing it to encode
    features useful for predicting BOTH direction AND risk level.
    Both heads are discarded after training — only the encoder trunk is kept.
    """

    def __init__(self, input_dim: int, latent_dim: int, n_classes: int = 3):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),       nn.ReLU(),
            nn.Linear(128, latent_dim), nn.Tanh(),
        )
        self.dir_head = nn.Linear(latent_dim, n_classes)       # direction logits
        self.vol_head = nn.Sequential(                          # volatility regression
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Softplus(),                   # always positive
        )

    def forward(self, x: "torch.Tensor"):
        z        = self.encoder(x)
        logits   = self.dir_head(z)
        vol_pred = self.vol_head(z).squeeze(-1)                # (batch,)
        return logits, vol_pred, z

    def encode(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.encoder(x)


class _TransformerEncoderNet(nn.Module if _TORCH_AVAILABLE else object):
    """Transformer encoder + classification head trained on direction labels.

    Drop-in replacement for _SupervisedEncoderNet. Instead of flattening the
    50-bar window into a vector and treating all bars equally, self-attention
    learns which bars within the window matter most for the prediction.

    Architecture: (batch, W*F flat) → reshape → (batch, W, F)
      → Linear(F→d_model) → TransformerEncoder(d_model, n_heads, n_layers)
      → CLS token → Linear(d_model→latent_dim) → Linear(latent_dim→3 classes)

    The CLS token aggregates sequence context; its representation after attention
    is used as the latent vector — analogous to BERT's [CLS] token.
    """

    def __init__(
        self,
        window_size: int,
        n_feats:     int,
        latent_dim:  int,
        d_model:     int = 32,
        n_heads:     int = 4,
        n_layers:    int = 2,
        n_classes:   int = 3,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        super().__init__()
        self._W = window_size
        self._F = n_feats
        self.input_proj = nn.Linear(n_feats, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.latent_head = nn.Linear(d_model, latent_dim)
        self.head        = nn.Linear(latent_dim, n_classes)

    def forward(self, x: "torch.Tensor"):
        B  = x.size(0)
        x  = x.view(B, self._W, self._F)                  # (B, W, F)
        x  = self.input_proj(x)                            # (B, W, d_model)
        cls = self.cls_token.expand(B, -1, -1)             # (B, 1, d_model)
        x  = torch.cat([cls, x], dim=1)                    # (B, W+1, d_model)
        x  = self.transformer(x)                           # (B, W+1, d_model)
        z  = self.latent_head(x[:, 0, :])                  # CLS → (B, latent_dim)
        logits = self.head(z)
        return logits, z

    def encode(self, x: "torch.Tensor") -> "torch.Tensor":
        _, z = self.forward(x)
        return z


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class LatentEncoder:
    """
    Decoupled latent feature extractor.

    Parameters
    ----------
    mode        : "supervised" (default) | "multitask" | "transformer" | "autoencoder"
                  supervised  — MLP + direction head (current best: Sharpe +3.13)
                  multitask   — MLP + direction head + volatility head (EXPERIMENT)
                  transformer — Attention enc + direction head (needs 150+ epochs + LR warmup)
                  autoencoder — Unsupervised MSE reconstruction (no labels needed)
    window_size    : OHLCV bars per input window (bars t-W … t-1)
    latent_dim     : size of the latent vector (8 recommended)
    epochs         : training epochs
    batch_size     : mini-batch size
    lr             : Adam learning rate
    random_state   : seed
    multitask_alpha      : weight of volatility loss (L = L_dir + alpha × L_vol)
    transformer_d_model  : Transformer hidden dim (only used when mode="transformer")
    transformer_n_heads  : Transformer attention heads (must divide d_model evenly)
    transformer_n_layers : Transformer encoder layers
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
        multitask_alpha:      float = 0.3,
        transformer_d_model:  int   = 32,
        transformer_n_heads:  int   = 4,
        transformer_n_layers: int   = 2,
    ):
        if mode not in ("supervised", "multitask", "transformer", "autoencoder"):
            raise ValueError(
                'mode must be "supervised", "multitask", "transformer", or "autoencoder"'
            )
        self.mode         = mode
        self.window_size  = window_size
        self.latent_dim   = latent_dim
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.random_state = random_state

        self.multitask_alpha      = multitask_alpha
        self.transformer_d_model  = transformer_d_model
        self.transformer_n_heads  = transformer_n_heads
        self.transformer_n_layers = transformer_n_layers

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

    def _compute_vol_target(self, close: "pd.Series", horizon: int = 4) -> "pd.Series":
        """Normalized absolute forward return — volatility proxy for multitask training.

        y_vol[t] = |close[t+h] - close[t]| / close[t], divided by expanding mean.
        Result > 1 means above-average volatility; < 1 means calm.
        Clipped to [0, 5] to prevent extreme outliers dominating the loss.
        """
        abs_ret  = (close.shift(-horizon) - close).abs() / close.clip(lower=1e-8)
        mean_abs = abs_ret.expanding(min_periods=20).mean().fillna(abs_ret.mean() or 1.0)
        return (abs_ret / (mean_abs + 1e-8)).clip(upper=5.0).fillna(1.0).astype(np.float32)

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
        if self.mode in ("supervised", "transformer", "multitask") and y is None:
            raise ValueError(
                f'mode="{self.mode}" requires labels y. '
                'Pass y=labels_series, or use mode="autoencoder".'
            )
        _require_torch()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        ohlcv_df  = self._extract_ohlcv(df)
        ohlcv     = ohlcv_df.values
        X         = self._build_windows(ohlcv)
        self._input_dim = X.shape[1]

        if self.mode in ("supervised", "transformer", "multitask"):
            y_vol = None
            if self.mode == "multitask":
                y_vol = self._compute_vol_target(ohlcv_df["close"], horizon=4)
            self._fit_supervised(X, ohlcv_df.index, y, y_vol=y_vol)
        else:
            self._fit_autoencoder(X)

        self._trained_on = str(df.index[0]) if hasattr(df.index, '__getitem__') else "unknown"
        return self

    def _fit_supervised(
        self,
        X: np.ndarray,
        ohlcv_index: "pd.Index",
        y: pd.Series,
        y_vol: Optional["pd.Series"] = None,
    ) -> None:
        """Train encoder + head(s) on direction labels (+ optional volatility for multitask)."""
        W = self.window_size
        bar_times = ohlcv_index[W:]
        labels    = y.reindex(bar_times)
        valid     = labels.notna()
        X_valid   = X[valid.values]
        y_valid   = labels[valid].values.astype(int)
        y_mapped  = np.array(
            [self._LABEL_MAP[int(l)] for l in y_valid], dtype=np.int64
        )

        # Volatility targets for multitask mode
        tensor_vol = None
        if self.mode == "multitask" and y_vol is not None:
            vol_aligned = y_vol.reindex(bar_times)
            vol_valid   = vol_aligned[valid.values].fillna(1.0).values.astype(np.float32)
            tensor_vol  = torch.from_numpy(vol_valid)

        if self.mode == "multitask":
            net = _MultiTaskEncoderNet(self._input_dim, self.latent_dim)
            print(f"[LatentEncoder] Architecture: MultiTask MLP  "
                  f"input_dim={self._input_dim}  alpha={self.multitask_alpha}", flush=True)
        elif self.mode == "transformer":
            n_feats = self._input_dim // self.window_size
            net = _TransformerEncoderNet(
                window_size = self.window_size,
                n_feats     = n_feats,
                latent_dim  = self.latent_dim,
                d_model     = self.transformer_d_model,
                n_heads     = self.transformer_n_heads,
                n_layers    = self.transformer_n_layers,
            )
            print(
                f"[LatentEncoder] Architecture: Transformer  "
                f"d_model={self.transformer_d_model}  "
                f"n_heads={self.transformer_n_heads}  "
                f"n_layers={self.transformer_n_layers}",
                flush=True,
            )
        else:
            net = _SupervisedEncoderNet(self._input_dim, self.latent_dim)
            print(f"[LatentEncoder] Architecture: MLP  input_dim={self._input_dim}", flush=True)

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

        criterion_vol = nn.MSELoss() if self.mode == "multitask" else None

        for epoch in range(1, self.epochs + 1):
            perm        = torch.randperm(n)
            epoch_loss  = 0.0
            correct     = 0
            for start in range(0, n, self.batch_size):
                idx    = perm[start : start + self.batch_size]
                bx, by = tensor_X[idx], tensor_y[idx]
                optimizer.zero_grad()

                if self.mode == "multitask":
                    logits, vol_pred, _ = net(bx)
                    bvol = tensor_vol[idx]
                    loss = criterion(logits, by) + self.multitask_alpha * criterion_vol(vol_pred, bvol)
                else:
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
                    "batch_size":        self.batch_size,
                    "lr":                self.lr,
                    "random_state":      self.random_state,
                    "multitask_alpha":   self.multitask_alpha,
                    "transformer_d_model":  self.transformer_d_model,
                    "transformer_n_heads":  self.transformer_n_heads,
                    "transformer_n_layers": self.transformer_n_layers,
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
        self.multitask_alpha      = p.get("multitask_alpha",      0.3)
        self.transformer_d_model  = p.get("transformer_d_model",  32)
        self.transformer_n_heads  = p.get("transformer_n_heads",  4)
        self.transformer_n_layers = p.get("transformer_n_layers", 2)
        self._trained_on  = ckpt.get("trained_on")

        if self.mode == "multitask":
            net = _MultiTaskEncoderNet(p["input_dim"], p["latent_dim"])
        elif self.mode == "transformer":
            n_feats = p["input_dim"] // p["window_size"]
            net = _TransformerEncoderNet(
                window_size = p["window_size"],
                n_feats     = n_feats,
                latent_dim  = p["latent_dim"],
                d_model     = self.transformer_d_model,
                n_heads     = self.transformer_n_heads,
                n_layers    = self.transformer_n_layers,
            )
        elif self.mode == "supervised":
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
