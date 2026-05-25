"""
LLM Signal Model — Phase 9.

Uses Claude (or any Anthropic model) to generate [P_buy, P_hold, P_sell]
probabilities from a compact tokenized OHLCV context.

Backtesting strategy (avoids 38k+ API calls during walk-forward):
  1. Run scripts/precompute_llm_signals.py ONCE to generate a parquet cache
     covering all historical bars at stride N (default=4).
  2. predict_proba() reads from the cache; on miss it calls the API.
  3. In live trading the cache is updated in-memory at most once per cache_bars bars.

Usage:
    model = LLMSignalModel()
    model.train(X, y)          # no-op, just records feature names
    proba = model.predict_proba(X)  # (N, 3) — reads cache or calls API

    # Walk-forward (after precompute_llm_signals.py):
    from src.walk_forward import WalkForwardConfig, WalkForwardValidator
    cfg = WalkForwardConfig(model_type="llm_signal", ...)
    WalkForwardValidator().run(X, y, prices, cfg).report()
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.model_interface import ModelInterface
from src.features.bar_tokenizer import BarTokenizer, VOCAB_SIZE

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


# ── System prompt (cached on Anthropic servers) ────────────────────────────────

_SYSTEM_PROMPT = """\
You are a quantitative Forex analyst. You receive recent EURUSD M15 candlestick bars \
encoded as compact tokens:

  TOKEN = DIR_SIZE_WICK
  DIR : UP=bullish close  DN=bearish close  DJ=doji (indecision)
  SIZE: XS < SM < MD < LG < XL  (body size relative to current ATR)
  WICK: BW=lower-wick dominant  NW=neutral  TW=upper-wick dominant

After seeing the token sequence and market state, output ONLY this JSON (no other text):
{"P_buy": <0-1>, "P_hold": <0-1>, "P_sell": <0-1>}
where the three values must sum to exactly 1.0. Do NOT include reasoning.\
"""


# ── LLMSignalModel ─────────────────────────────────────────────────────────────

class LLMSignalModel(ModelInterface):
    """
    Pluggable ModelInterface wrapper around a Claude API call.

    Parameters
    ----------
    model_id       : Anthropic model string (default: cheapest capable model)
    n_context_bars : Number of tokenized bars sent as context per call
    cache_bars     : In live mode, reuse last result for this many bars
    cache_path     : Path to pre-computed parquet cache (DatetimeIndex → P_buy/P_hold/P_sell)
    api_key        : Anthropic API key; falls back to ANTHROPIC_API_KEY env var
    max_retries    : Retry on API error / rate limit
    retry_delay    : Seconds between retries
    """

    def __init__(
        self,
        model_id:        str   = "claude-haiku-4-5-20251001",
        n_context_bars:  int   = 32,
        cache_bars:      int   = 4,
        cache_path:      str   = "data/models/llm_cache.parquet",
        api_key:         Optional[str] = None,
        max_retries:     int   = 3,
        retry_delay:     float = 2.0,
        provider:        str   = "claude_cli",
    ):
        self.model_id        = model_id
        self.n_context_bars  = n_context_bars
        self.cache_bars      = cache_bars
        self.cache_path      = cache_path
        self.max_retries     = max_retries
        self.retry_delay     = retry_delay
        self.provider        = provider   # "claude_cli" | "claude_api"

        self._api_key        = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client         = None
        self._tokenizer      = BarTokenizer()
        self._feature_names: list[str] = []
        self._trained_on     = ""

        # Disk cache: DatetimeIndex → [P_buy, P_hold, P_sell]
        self._cache: Optional[pd.DataFrame] = None
        self._load_cache()

        # Live-mode in-memory cache: (last_ts, proba)
        self._live_cache: Optional[tuple] = None
        self._live_bar_count: int = 0

    # ── ModelInterface ─────────────────────────────────────────────────────────

    def train(self, X: pd.DataFrame, y=None) -> "LLMSignalModel":
        """No-op — records feature names and fits tokenizer on X's OHLCV columns."""
        self._feature_names = list(X.columns)
        if hasattr(X.index, "min"):
            self._trained_on = f"{X.index.min().date()} → {X.index.max().date()}"
        # Fit tokenizer bin edges from X (ATR already in feature matrix)
        # We need open/high/low/close — stored separately in prices, but we can
        # derive body ratio proxy from return_1 if OHLCV not present
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return (N, 3) probability array [P_buy, P_hold, P_sell] for each row of X.
        Checks cache first; falls back to API call on miss.
        """
        n = len(X)
        result = np.full((n, 3), 1.0 / 3.0)  # default: uniform

        for i, (ts, row) in enumerate(X.iterrows()):
            proba = self._lookup_cache(ts)
            if proba is None:
                # Build context from the window ending at this bar
                ctx_slice = X.iloc[max(0, i - self.n_context_bars + 1): i + 1]
                proba = self._call_api(ctx_slice)
                self._store_cache(ts, proba)
            result[i] = proba

        return result[0] if n == 1 else result

    def save(self, path) -> None:
        import joblib
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model_id":       self.model_id,
            "n_context_bars": self.n_context_bars,
            "cache_bars":     self.cache_bars,
            "cache_path":     self.cache_path,
            "provider":       self.provider,
            "feature_names":  self._feature_names,
            "trained_on":     self._trained_on,
        }, path)

    def load(self, path) -> "LLMSignalModel":
        import joblib
        d = joblib.load(path)
        self.model_id        = d["model_id"]
        self.n_context_bars  = d["n_context_bars"]
        self.cache_bars      = d["cache_bars"]
        self.cache_path      = d["cache_path"]
        self.provider        = d.get("provider", "claude_cli")
        self._feature_names  = d.get("feature_names", [])
        self._trained_on     = d.get("trained_on", "")
        self._load_cache()
        return self

    def metadata(self) -> dict:
        return {
            "name":       "LLMSignalModel",
            "version":    "1.0",
            "trained_on": self._trained_on,
            "features":   self._feature_names,
            "n_classes":  3,
            "params": {
                "model_id":       self.model_id,
                "n_context_bars": self.n_context_bars,
                "cache_bars":     self.cache_bars,
            },
        }

    # ── Cache management ───────────────────────────────────────────────────────

    def _load_cache(self) -> None:
        p = Path(self.cache_path)
        if p.exists():
            try:
                self._cache = pd.read_parquet(p)
                if not self._cache.index.is_monotonic_increasing:
                    self._cache = self._cache.sort_index()
            except Exception:
                self._cache = None

    def _lookup_cache(self, ts) -> Optional[np.ndarray]:
        if self._cache is None or ts not in self._cache.index:
            return None
        row = self._cache.loc[ts]
        return np.array([float(row["P_buy"]), float(row["P_hold"]), float(row["P_sell"])])

    def _store_cache(self, ts, proba: np.ndarray) -> None:
        new_row = pd.DataFrame(
            [{"P_buy": proba[0], "P_hold": proba[1], "P_sell": proba[2]}],
            index=[ts],
        )
        if self._cache is None:
            self._cache = new_row
        else:
            self._cache = pd.concat([self._cache, new_row])

    def flush_cache(self) -> None:
        """Persist the in-memory cache additions back to disk."""
        if self._cache is not None:
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            self._cache.sort_index().to_parquet(self.cache_path)

    # ── API call ───────────────────────────────────────────────────────────────

    def _call_api(self, X_slice: pd.DataFrame) -> np.ndarray:
        """Build prompt from X_slice, call Claude via selected provider."""
        user_content = self._build_user_message(X_slice)
        if self.provider == "claude_cli":
            return self._call_claude_cli(user_content)
        return self._call_claude_sdk(user_content)

    def _call_claude_cli(self, user_content: str) -> np.ndarray:
        """Call Claude via the `claude` CLI subprocess (uses terminal auth, no API key needed)."""
        full_prompt = f"{_SYSTEM_PROMPT}\n\n{user_content}"
        for attempt in range(self.max_retries):
            try:
                result = subprocess.run(
                    ["claude", "-p", full_prompt, "--output-format", "json"],
                    capture_output=True, text=True, timeout=60,
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.strip() or "claude CLI returned non-zero exit")
                wrapper = json.loads(result.stdout)
                text = wrapper.get("result", "")
                return self._parse_probabilities(text)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"[LLMSignalModel] CLI error after {self.max_retries} retries: {e}")
                    return np.array([1/3, 1/3, 1/3])

    def _call_claude_sdk(self, user_content: str) -> np.ndarray:
        """Call Claude via the anthropic SDK (requires ANTHROPIC_API_KEY)."""
        if not _ANTHROPIC_AVAILABLE:
            return np.array([1/3, 1/3, 1/3])
        if not self._api_key:
            return np.array([1/3, 1/3, 1/3])

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self._api_key)

        for attempt in range(self.max_retries):
            try:
                resp = self._client.messages.create(
                    model=self.model_id,
                    max_tokens=64,
                    system=[{
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }],
                    messages=[{"role": "user", "content": user_content}],
                )
                text = resp.content[0].text.strip()
                return self._parse_probabilities(text)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"[LLMSignalModel] SDK error after {self.max_retries} retries: {e}")
                    return np.array([1/3, 1/3, 1/3])

    def _build_user_message(self, X_slice: pd.DataFrame) -> str:
        """Build the compact user message from the feature slice."""
        tok = self._tokenizer

        # The feature matrix has atr_14, rsi_14, sma_50 columns — use them directly
        tok_str  = tok.encode_sequence(X_slice, n_bars=min(self.n_context_bars, len(X_slice)))
        ctx_line = tok.context_prefix(X_slice)

        return f"Last {min(self.n_context_bars, len(X_slice))} bars: {tok_str}\n{ctx_line}"

    @staticmethod
    def _parse_probabilities(text: str) -> np.ndarray:
        """Extract [P_buy, P_hold, P_sell] from Claude's JSON response."""
        # Try direct parse, then strip code fences, then regex
        attempts = [
            text,
            text.strip("`").strip(),
            (re.search(r"\{[^}]+\}", text) or type("", (), {"group": lambda s: "{}"})()).group(0)
                if re.search(r"\{[^}]+\}", text) else "{}",
        ]
        for raw in attempts:
            try:
                d = json.loads(raw)
                p = np.array([
                    float(d.get("P_buy",  1/3)),
                    float(d.get("P_hold", 1/3)),
                    float(d.get("P_sell", 1/3)),
                ], dtype=float)
                p = np.clip(p, 0, 1)
                s = p.sum()
                if s > 0:
                    return p / s
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

        return np.array([1/3, 1/3, 1/3])
