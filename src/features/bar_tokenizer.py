"""
OHLCV Bar Tokenizer — shared encoding layer for LLM and local LM models.

Each candlestick bar is mapped to a compact 3-part symbol:
    TOKEN = DIR_SIZE_WICK
    DIR : UP | DN | DJ  (up / down / doji when |body/ATR| < doji_threshold)
    SIZE: XS | SM | MD | LG | XL  (body size in ATR multiples, 5 quantile bins)
    WICK: BW | NW | TW  (bottom-wick dominant / neutral / top-wick dominant)

Vocabulary: 3 × 5 × 3 = 45 bar tokens + PAD (id=0) + UNK (id=1) = 47 total.

Two output formats:
    encode_sequence(df, n_bars) → str   for LLM text prompts
    encode_ids(df, n_bars)      → array for local Transformer model

Usage:
    tok = BarTokenizer()
    tok.fit(prices_df)                              # learn SIZE bin edges from data
    text = tok.encode_sequence(df_with_atr, n_bars=32)
    ids  = tok.encode_ids(df_with_atr, n_bars=32)
    joblib.dump(tok, "data/models/bar_tokenizer.joblib")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ── Constants ──────────────────────────────────────────────────────────────────

PAD_TOKEN = "PAD"
UNK_TOKEN = "UNK"

_DIRS  = ("UP", "DN", "DJ")
_SIZES = ("XS", "SM", "MD", "LG", "XL")
_WICKS = ("BW", "NW", "TW")

# Build vocabulary: all valid token strings + PAD + UNK
_BAR_TOKENS = [f"{d}_{s}_{w}" for d in _DIRS for s in _SIZES for w in _WICKS]
_ALL_TOKENS  = [PAD_TOKEN, UNK_TOKEN] + _BAR_TOKENS   # PAD=0, UNK=1, bars 2…46

VOCAB_SIZE   = len(_ALL_TOKENS)          # 47
PAD_ID       = 0
UNK_ID       = 1


# ── BarTokenizer ───────────────────────────────────────────────────────────────

class BarTokenizer:
    """
    Fit once on historical data to learn SIZE bin edges, then encode any
    OHLCV DataFrame into either text tokens or integer ids.

    Parameters
    ----------
    atr_col       : Column name of pre-computed ATR in the feature matrix.
                    If absent the tokenizer falls back to a rolling high-low range.
    rsi_col       : Column for context_prefix() market state summary.
    sma_slow_col  : Column for trend direction in context_prefix().
    doji_threshold: |body/ATR| below this → DJ (doji) direction.
    wick_threshold: wick/body ratio above this → wick is "dominant".
    n_body_bins   : Number of SIZE quantile bins (default 5).
    """

    def __init__(
        self,
        atr_col:        str   = "atr_14",
        rsi_col:        str   = "rsi_14",
        sma_slow_col:   str   = "sma_50",
        doji_threshold: float = 0.10,
        wick_threshold: float = 0.40,
        n_body_bins:    int   = 5,
    ):
        self.atr_col        = atr_col
        self.rsi_col        = rsi_col
        self.sma_slow_col   = sma_slow_col
        self.doji_threshold = doji_threshold
        self.wick_threshold = wick_threshold
        self.n_body_bins    = n_body_bins

        # Built by fit()
        self._bin_edges: Optional[np.ndarray] = None  # shape (n_body_bins - 1,)
        self._is_fitted = False

        # Vocabulary lookups
        self.vocab:     dict[str, int] = {t: i for i, t in enumerate(_ALL_TOKENS)}
        self.inv_vocab: dict[int, str] = {i: t for t, i in self.vocab.items()}

    # ── Fit ────────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "BarTokenizer":
        """
        Learn SIZE quantile bin edges from the body/ATR distribution of df.
        df must have OHLCV columns (open, high, low, close) and ideally
        the atr column.  Call once on training data, then transform test data.
        """
        atr = self._get_atr(df)
        body = (df["close"] - df["open"]).abs()
        ratio = (body / atr.replace(0, np.nan)).dropna()
        quantiles = np.linspace(0, 1, self.n_body_bins + 1)[1:-1]  # inner edges
        self._bin_edges = np.quantile(ratio, quantiles)
        self._is_fitted  = True
        return self

    # ── Encoding ───────────────────────────────────────────────────────────────

    def encode_bar(
        self,
        open_: float,
        high:  float,
        low:   float,
        close: float,
        atr:   float,
    ) -> str:
        """Single bar → token string e.g. 'UP_MD_TW'."""
        if atr <= 0:
            return UNK_TOKEN

        body       = close - open_
        body_abs   = abs(body)
        body_ratio = body_abs / atr

        # Direction
        if body_ratio < self.doji_threshold:
            direction = "DJ"
        elif body > 0:
            direction = "UP"
        else:
            direction = "DN"

        # Size (quantile bin)
        size = self._body_ratio_to_size(body_ratio)

        # Wick dominance
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        candle_range = high - low if high > low else atr
        upper_rel = upper_wick / candle_range
        lower_rel = lower_wick / candle_range

        if upper_rel > self.wick_threshold and upper_rel > lower_rel * 1.5:
            wick = "TW"
        elif lower_rel > self.wick_threshold and lower_rel > upper_rel * 1.5:
            wick = "BW"
        else:
            wick = "NW"

        return f"{direction}_{size}_{wick}"

    def encode_sequence(self, df: pd.DataFrame, n_bars: int = 32) -> str:
        """
        Last n_bars of df → space-separated token string for LLM prompts.
        df must have: open, high, low, close.  Optionally: atr_col column.
        """
        slice_ = df.tail(n_bars)
        atr    = self._get_atr(slice_)
        tokens = []
        for i, (idx, row) in enumerate(slice_.iterrows()):
            a = float(atr.iloc[i]) if hasattr(atr, "iloc") else float(atr)
            tokens.append(self.encode_bar(
                float(row["open"]), float(row["high"]),
                float(row["low"]),  float(row["close"]), a
            ))
        return " ".join(tokens)

    def encode_ids(self, df: pd.DataFrame, n_bars: int = 32) -> np.ndarray:
        """
        Last n_bars of df → integer id array, shape (n_bars,).
        Unknown tokens → UNK_ID.  Short sequences → left-padded with PAD_ID.
        """
        slice_ = df.tail(n_bars)
        atr    = self._get_atr(slice_)
        ids    = []
        for i, (idx, row) in enumerate(slice_.iterrows()):
            a   = float(atr.iloc[i]) if hasattr(atr, "iloc") else float(atr)
            tok = self.encode_bar(
                float(row["open"]), float(row["high"]),
                float(row["low"]),  float(row["close"]), a
            )
            ids.append(self.vocab.get(tok, UNK_ID))

        # Left-pad if fewer bars than requested
        if len(ids) < n_bars:
            ids = [PAD_ID] * (n_bars - len(ids)) + ids

        return np.array(ids, dtype=np.int64)

    def encode_all_ids(self, df: pd.DataFrame, seq_len: int = 32) -> np.ndarray:
        """
        Encode entire df bar-by-bar into integer ids (shape: len(df),).
        Used by BarLMModel to build sliding-window sequences.
        """
        atr = self._get_atr(df)
        ids = []
        for i, (idx, row) in enumerate(df.iterrows()):
            a   = float(atr.iloc[i])
            tok = self.encode_bar(
                float(row["open"]), float(row["high"]),
                float(row["low"]),  float(row["close"]), a
            )
            ids.append(self.vocab.get(tok, UNK_ID))
        return np.array(ids, dtype=np.int64)

    def context_prefix(self, df: pd.DataFrame) -> str:
        """
        2-line market state summary appended below the token sequence in LLM prompts.
        Uses the last available row's indicator values.
        """
        row  = df.iloc[-1]
        atr  = self._get_atr(df)
        atr_val  = float(atr.iloc[-1])
        atr_pips = atr_val / 0.0001  # EURUSD default (1 pip = 0.0001)
        # Sanity-cap: if atr_pips is unreasonable (e.g. scaled feature snuck in),
        # fall back to H-L range of last bar
        if not (0.1 <= atr_pips <= 500):
            if "high" in df.columns and "low" in df.columns:
                atr_pips = (float(row["high"]) - float(row["low"])) / 0.0001
            else:
                atr_pips = 8.0  # sensible EURUSD default

        parts = [f"ATR: {atr_pips:.1f} pips"]

        if self.rsi_col in df.columns:
            rsi = float(row[self.rsi_col])
            # If RSI looks scaled (outside 0-100), skip it
            if 0 <= rsi <= 100:
                label = "overbought" if rsi > 70 else ("oversold" if rsi < 30 else "neutral")
                parts.append(f"RSI: {rsi:.0f} ({label})")

        if self.sma_slow_col in df.columns and "close" in df.columns:
            close = float(row["close"])
            sma   = float(row[self.sma_slow_col])
            # Only show trend if both look like raw prices (positive, similar magnitude)
            if close > 0 and sma > 0 and 0.5 < close / sma < 2.0:
                trend = "BULLISH" if close > sma else "BEARISH"
                parts.append(f"Trend: {trend} (price {'above' if close > sma else 'below'} SMA50)")

        return " | ".join(parts)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "bin_edges":      self._bin_edges,
            "is_fitted":      self._is_fitted,
            "atr_col":        self.atr_col,
            "rsi_col":        self.rsi_col,
            "sma_slow_col":   self.sma_slow_col,
            "doji_threshold": self.doji_threshold,
            "wick_threshold": self.wick_threshold,
            "n_body_bins":    self.n_body_bins,
        }, path)

    @classmethod
    def load(cls, path: str) -> "BarTokenizer":
        import joblib
        d = joblib.load(path)
        tok = cls(
            atr_col        = d["atr_col"],
            rsi_col        = d["rsi_col"],
            sma_slow_col   = d["sma_slow_col"],
            doji_threshold = d["doji_threshold"],
            wick_threshold = d["wick_threshold"],
            n_body_bins    = d["n_body_bins"],
        )
        tok._bin_edges = d["bin_edges"]
        tok._is_fitted = d["is_fitted"]
        return tok

    # ── Internals ──────────────────────────────────────────────────────────────

    def _body_ratio_to_size(self, ratio: float) -> str:
        if not self._is_fitted or self._bin_edges is None:
            # Fallback: fixed thresholds in ATR multiples
            edges = np.array([0.1, 0.3, 0.6, 1.0])
        else:
            edges = self._bin_edges
        idx = int(np.searchsorted(edges, ratio, side="right"))
        return _SIZES[min(idx, len(_SIZES) - 1)]

    def _get_atr(self, df: pd.DataFrame) -> pd.Series:
        """
        Return ATR series from df (raw price units, always positive).
        Priority: H-L range (raw prices) > atr_col if its median looks raw > return_1 proxy.
        Never uses a StandardScaler-normalised feature column directly — scaled ATR
        is centred at 0 and half the values would be negative.
        """
        # Best: compute directly from raw H-L prices
        if "high" in df.columns and "low" in df.columns:
            hl = (df["high"] - df["low"]).clip(lower=1e-8)
            return hl.rolling(14, min_periods=1).mean()

        # Use precomputed atr_col only when values look raw (median > 1e-5 for EURUSD)
        if self.atr_col in df.columns:
            col = df[self.atr_col].fillna(method="ffill")
            if float(col.median()) > 1e-5:
                return col.clip(lower=1e-8)

        # Final fallback: absolute 1-bar return as rough ATR proxy
        if "return_1" in df.columns:
            return (df["return_1"].abs()
                    .rolling(14, min_periods=1).mean()
                    .clip(lower=1e-8))

        return pd.Series(1e-4, index=df.index)

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def vocab_table(self) -> None:
        """Print the full vocabulary with ids."""
        print(f"\n{'ID':>4}  {'Token':<15}  {'Type'}")
        print("─" * 34)
        for tok, tid in self.vocab.items():
            kind = "special" if tid < 2 else "bar"
            print(f"  {tid:>2}  {tok:<15}  {kind}")

    def distribution(self, df: pd.DataFrame) -> dict:
        """Return token frequency dict for df."""
        from collections import Counter
        atr = self._get_atr(df)
        toks = [
            self.encode_bar(
                float(row["open"]), float(row["high"]),
                float(row["low"]),  float(row["close"]),
                float(atr.iloc[i])
            )
            for i, (_, row) in enumerate(df.iterrows())
        ]
        return dict(Counter(toks))
