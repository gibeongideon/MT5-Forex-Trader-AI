"""
Suffix Automaton + Autoencoder Position Sizer — Python port of CMoneySuffixAE.

Proactive lot-size scaling based on current market structure rather than
account trade history.  Two engines run in sequence:

  1. SuffixAutomaton  — scores how familiar the current price-DNA sequence
                        is against the full historical pattern index.
  2. AutoEncoder      — measures structural integrity via reconstruction error;
                        a high MSE means the market's geometry is breaking.

The combined multiplier is applied on top of whatever lot size the caller
starts with.  Plug into RiskManager as an optional post-sizing layer.

Usage:
    sizer = SuffixAESizer(history_length=500, dna_window=16, algo_mode=1, use_ae=True)
    multiplier = sizer.compute(closes)          # closes[0] = most recent bar
    sized_lots = base_lots * multiplier
"""

from __future__ import annotations

import math
from typing import Sequence


# ── Suffix Automaton ────────────────────────────────────────────────────────────

class _SAState:
    __slots__ = ("len", "link", "next")

    def __init__(self) -> None:
        self.len: int = 0
        self.link: int = -1
        self.next: dict[int, int] = {}


class SuffixAutomaton:
    """
    Builds a suffix automaton over an integer alphabet (0=U, 1=D, 2=F).
    Construction is O(n); each query is O(|T|).
    """

    def __init__(self) -> None:
        self._states: list[_SAState] = []
        self._last: int = 0
        self.reset()

    def reset(self) -> None:
        self._states = [_SAState()]   # initial state at index 0
        self._states[0].len = 0
        self._states[0].link = -1
        self._last = 0

    def extend(self, c: int) -> None:
        """Append one character from {0, 1, 2} to the automaton."""
        cur = len(self._states)
        s = _SAState()
        s.len = self._states[self._last].len + 1
        self._states.append(s)

        p = self._last
        while p != -1 and c not in self._states[p].next:
            self._states[p].next[c] = cur
            p = self._states[p].link

        if p == -1:
            s.link = 0
        else:
            q = self._states[p].next[c]
            if self._states[p].len + 1 == self._states[q].len:
                s.link = q
            else:
                clone = len(self._states)
                cl = _SAState()
                cl.len = self._states[p].len + 1
                cl.link = self._states[q].link
                cl.next = dict(self._states[q].next)
                self._states.append(cl)

                while p != -1 and self._states[p].next.get(c) == q:
                    self._states[p].next[c] = clone
                    p = self._states[p].link

                self._states[q].link = clone
                s.link = clone

        self._last = cur

    def get_longest_match(self, pattern: Sequence[int]) -> int:
        """
        Return the length of the longest prefix of `pattern` that appears
        anywhere in the string indexed by this automaton.
        """
        cur = 0
        matched = 0
        for c in pattern:
            while cur != 0 and c not in self._states[cur].next:
                cur = self._states[cur].link
                matched = self._states[cur].len
            if c in self._states[cur].next:
                cur = self._states[cur].next[c]
                matched += 1
        return matched


# ── Autoencoder (numpy-free, pure Python) ───────────────────────────────────────

def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


class AutoEncoder:
    """
    Lightweight single-hidden-layer autoencoder trained online via one pass
    of gradient descent each time Calculate() is called.

    Architecture:  input(n) → hidden(k) → output(n)
    Loss:          MSE between input and reconstruction
    Confidence:    C = 1 / (1 + MSE)   maps loss ∈ [0,∞) → C ∈ (0,1]
    """

    def __init__(self, input_size: int = 16, hidden_size: int = 4,
                 lr: float = 0.05, train_steps: int = 20) -> None:
        self.n = input_size
        self.k = hidden_size
        self.lr = lr
        self.train_steps = train_steps

        # Xavier-like weight init (deterministic: all 0.1)
        self._we: list[list[float]] = [[0.1] * input_size for _ in range(hidden_size)]
        self._be: list[float] = [0.0] * hidden_size
        self._wd: list[list[float]] = [[0.1] * hidden_size for _ in range(input_size)]
        self._bd: list[float] = [0.0] * input_size

    def _encode(self, x: list[float]) -> list[float]:
        return [_sigmoid(sum(self._we[j][i] * x[i] for i in range(self.n)) + self._be[j])
                for j in range(self.k)]

    def _decode(self, h: list[float]) -> list[float]:
        return [_sigmoid(sum(self._wd[i][j] * h[j] for j in range(self.k)) + self._bd[i])
                for i in range(self.n)]

    def _mse(self, x: list[float], x_hat: list[float]) -> float:
        return sum((a - b) ** 2 for a, b in zip(x, x_hat)) / len(x)

    def _train_step(self, x: list[float]) -> None:
        h = self._encode(x)
        x_hat = self._decode(h)

        # Output layer gradients
        d_out = [(x_hat[i] - x[i]) * x_hat[i] * (1 - x_hat[i]) for i in range(self.n)]

        # Hidden layer gradients
        d_hid = [
            sum(self._wd[i][j] * d_out[i] for i in range(self.n)) * h[j] * (1 - h[j])
            for j in range(self.k)
        ]

        # Update decoder weights
        for i in range(self.n):
            for j in range(self.k):
                self._wd[i][j] -= self.lr * d_out[i] * h[j]
            self._bd[i] -= self.lr * d_out[i]

        # Update encoder weights
        for j in range(self.k):
            for i in range(self.n):
                self._we[j][i] -= self.lr * d_hid[j] * x[i]
            self._be[j] -= self.lr * d_hid[j]

    def calculate(self, x: list[float]) -> float:
        """
        Train briefly on x, then return confidence coefficient C = 1/(1+MSE).
        C ≈ 1.0 → low reconstruction error → structure intact.
        C → 0.0 → high reconstruction error → geometry breaking.
        """
        for _ in range(self.train_steps):
            self._train_step(x)
        h = self._encode(x)
        x_hat = self._decode(h)
        mse = self._mse(x, x_hat)
        return 1.0 / (1.0 + mse)


# ── Algorithm Modes ─────────────────────────────────────────────────────────────

def _mode1_linear(score: float) -> float:
    return 0.8 + 1.2 * score


def _mode2_conservative(score: float) -> float:
    return 0.5 + math.sqrt(score) * 0.75


def _mode3_aggressive(score: float) -> float:
    return 0.5 + 2.5 * score * score


def _mode4_mean_reversion(score: float) -> float:
    if score <= 0.0:
        return 1.0
    return min(3.0, 0.5 / score)


_MODES = {1: _mode1_linear, 2: _mode2_conservative,
          3: _mode3_aggressive, 4: _mode4_mean_reversion}


# ── Public sizer ────────────────────────────────────────────────────────────────

class SuffixAESizer:
    """
    Drop-in lot-size multiplier based on price-pattern familiarity and
    structural integrity.

    Parameters
    ----------
    history_length : bars used to build the historical DNA index
    dna_window     : bars used as the "recent setup" query
    algo_mode      : 1=linear, 2=conservative, 3=aggressive, 4=mean-reversion
    use_ae         : whether to apply the autoencoder gate
    ae_hidden      : hidden units in the autoencoder
    ae_lr          : learning rate for the online AE training
    ae_steps       : gradient steps per call (more = better fit, slower)
    """

    def __init__(
        self,
        history_length: int = 300,
        dna_window: int = 16,
        algo_mode: int = 1,
        use_ae: bool = True,
        ae_hidden: int = 4,
        ae_lr: float = 0.05,
        ae_steps: int = 20,
    ) -> None:
        if algo_mode not in _MODES:
            raise ValueError(f"algo_mode must be 1-4, got {algo_mode}")
        self.history_length = history_length
        self.dna_window = dna_window
        self.algo_mode = algo_mode
        self.use_ae = use_ae
        self._sa = SuffixAutomaton()
        self._ae = AutoEncoder(dna_window, ae_hidden, ae_lr, ae_steps)

    def compute(self, closes: Sequence[float]) -> float:
        """
        Compute lot-size multiplier from recent close prices.

        Parameters
        ----------
        closes : sequence of closing prices, closes[0] = most recent bar.
                 Must have at least history_length + dna_window elements.

        Returns
        -------
        multiplier ∈ (0, ~3]  — multiply your base lot size by this value.
        Returns 1.0 if not enough data.
        """
        needed = self.history_length + self.dna_window
        if len(closes) < needed:
            return 1.0

        # ── Step 1: Build the historical Suffix Automaton ─────────────────
        self._sa.reset()
        for i in range(self.dna_window, self.history_length + self.dna_window - 1):
            diff = closes[i] - closes[i + 1]
            char_code = 0 if diff > 0 else (1 if diff < 0 else 2)
            self._sa.extend(char_code)

        # ── Step 2: Extract recent DNA + raw price window ─────────────────
        recent_dna: list[int] = []
        recent_raw: list[float] = []
        max_p = closes[0]
        min_p = closes[0]

        for i in range(self.dna_window):
            diff = closes[self.dna_window - 1 - i] - closes[self.dna_window - i]
            recent_dna.append(0 if diff > 0 else (1 if diff < 0 else 2))
            p = closes[self.dna_window - 1 - i]
            recent_raw.append(p)
            if p > max_p:
                max_p = p
            if p < min_p:
                min_p = p

        # ── Step 3: Query automaton → dna_score → scale_factor ───────────
        longest_match = self._sa.get_longest_match(recent_dna)
        dna_score = longest_match / self.dna_window
        scale_factor = _MODES[self.algo_mode](dna_score)

        # ── Step 4: Autoencoder structural gate ───────────────────────────
        ae_coeff = 1.0
        if self.use_ae:
            norm_range = (max_p - min_p) if (max_p - min_p) != 0 else 1.0
            norm_raw = [(v - min_p) / norm_range for v in recent_raw]
            ae_coeff = self._ae.calculate(norm_raw)

        return round(scale_factor * ae_coeff, 4)

    def describe(self, closes: Sequence[float]) -> None:
        """Print a diagnostic breakdown (useful during backtesting)."""
        needed = self.history_length + self.dna_window
        if len(closes) < needed:
            print(f"  [SuffixAE] Not enough data (need {needed}, got {len(closes)})")
            return

        self._sa.reset()
        for i in range(self.dna_window, self.history_length + self.dna_window - 1):
            diff = closes[i] - closes[i + 1]
            self._sa.extend(0 if diff > 0 else (1 if diff < 0 else 2))

        recent_dna: list[int] = []
        recent_raw: list[float] = []
        max_p = min_p = closes[0]

        for i in range(self.dna_window):
            diff = closes[self.dna_window - 1 - i] - closes[self.dna_window - i]
            recent_dna.append(0 if diff > 0 else (1 if diff < 0 else 2))
            p = closes[self.dna_window - 1 - i]
            recent_raw.append(p)
            max_p = max(max_p, p)
            min_p = min(min_p, p)

        match = self._sa.get_longest_match(recent_dna)
        dna_score = match / self.dna_window
        scale = _MODES[self.algo_mode](dna_score)

        labels = {0: "U", 1: "D", 2: "F"}
        dna_str = "".join(labels[c] for c in recent_dna)

        norm_range = (max_p - min_p) if (max_p - min_p) != 0 else 1.0
        norm_raw = [(v - min_p) / norm_range for v in recent_raw]
        ae_c = self._ae.calculate(norm_raw) if self.use_ae else None

        sep = "─" * 48
        print(f"\n{sep}")
        print("  SuffixAE Position Sizer")
        print(sep)
        print(f"  Mode        : {self.algo_mode}  |  AE gate: {'on' if self.use_ae else 'off'}")
        print(f"  Price DNA   : {dna_str}")
        print(f"  Match len   : {match}/{self.dna_window}  ({dna_score:.0%})")
        print(f"  Scale factor: {scale:.4f}")
        if ae_c is not None:
            print(f"  AE coeff    : {ae_c:.4f}")
        print(f"  Multiplier  : {round(scale * (ae_c if ae_c else 1.0), 4):.4f}")
        print(sep + "\n")
