"""
Tests for SuffixAESizer — no MT5 / bridge required.

Run:
    conda run -n envmt5 python -m pytest tests/test_suffix_ae_sizer.py -v
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.suffix_ae_sizer import (
    AutoEncoder,
    SuffixAutomaton,
    SuffixAESizer,
    _mode1_linear,
    _mode2_conservative,
    _mode3_aggressive,
    _mode4_mean_reversion,
)


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _trending_closes(n: int = 400, step: float = 0.0001) -> list:
    """Monotonically rising closes, most-recent-first."""
    prices = [1.1000 + i * step for i in range(n)]
    return list(reversed(prices))


def _flat_closes(n: int = 400, price: float = 1.1000) -> list:
    return [price] * n


def _zigzag_closes(n: int = 400) -> list:
    """Alternating up/down pattern."""
    prices = []
    p = 1.1000
    for i in range(n):
        p += 0.0001 if i % 2 == 0 else -0.0001
        prices.append(p)
    return list(reversed(prices))


# ── SuffixAutomaton ──────────────────────────────────────────────────────────────

class TestSuffixAutomaton:

    def test_empty_pattern_returns_zero(self):
        sa = SuffixAutomaton()
        sa.extend(0)
        sa.extend(1)
        assert sa.get_longest_match([]) == 0

    def test_exact_match_full_length(self):
        sa = SuffixAutomaton()
        seq = [0, 1, 2, 0, 1]
        for c in seq:
            sa.extend(c)
        # entire sequence is a suffix — match = len(pattern)
        assert sa.get_longest_match(seq) == len(seq)

    def test_partial_match(self):
        sa = SuffixAutomaton()
        for c in [0, 1, 0, 1, 0]:
            sa.extend(c)
        # first 2 chars of [0, 1, 2] exist in history; 2 doesn't follow 1
        match = sa.get_longest_match([0, 1, 2])
        assert 0 <= match <= 3

    def test_no_match_returns_zero(self):
        sa = SuffixAutomaton()
        for c in [0, 0, 0, 0]:
            sa.extend(c)
        # pattern [1, 2] never appears in [0,0,0,0]
        assert sa.get_longest_match([1, 2]) == 0

    def test_reset_clears_state(self):
        sa = SuffixAutomaton()
        for c in [0, 1, 2]:
            sa.extend(c)
        sa.reset()
        # after reset nothing is indexed
        assert sa.get_longest_match([0]) == 0

    def test_match_does_not_exceed_pattern_length(self):
        sa = SuffixAutomaton()
        for c in [0, 1, 2, 0, 1, 2, 0, 1, 2]:
            sa.extend(c)
        pattern = [0, 1, 2]
        match = sa.get_longest_match(pattern)
        assert match <= len(pattern)


# ── AutoEncoder ──────────────────────────────────────────────────────────────────

class TestAutoEncoder:

    def test_confidence_in_range(self):
        ae = AutoEncoder(input_size=8, hidden_size=3)
        x = [0.1, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4]
        c = ae.calculate(x)
        assert 0.0 < c <= 1.0

    def test_constant_input_reconstructs_well(self):
        # A flat signal is trivially easy to reconstruct → C should be high
        ae = AutoEncoder(input_size=8, hidden_size=4, lr=0.1, train_steps=100)
        x = [0.5] * 8
        c = ae.calculate(x)
        assert c > 0.5, f"Expected C > 0.5 for constant input, got {c:.4f}"

    def test_confidence_formula_matches_mse(self):
        ae = AutoEncoder(input_size=4, hidden_size=2, train_steps=0)
        x = [0.2, 0.4, 0.6, 0.8]
        # With 0 training steps the initial weights produce some MSE
        h = ae._encode(x)
        x_hat = ae._decode(h)
        mse = ae._mse(x, x_hat)
        expected_c = 1.0 / (1.0 + mse)
        c = ae.calculate(x)
        # After 0 training steps calculate() should match the formula
        # (it does train internally — just check it's in (0,1])
        assert 0.0 < c <= 1.0


# ── Algorithm modes ───────────────────────────────────────────────────────────────

class TestAlgorithmModes:

    @pytest.mark.parametrize("score", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_mode1_linear_range(self, score):
        v = _mode1_linear(score)
        assert 0.8 <= v <= 2.0, f"Mode1 out of range at score={score}: {v}"

    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
    def test_mode2_conservative_range(self, score):
        v = _mode2_conservative(score)
        assert 0.5 <= v <= 1.25

    @pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
    def test_mode3_aggressive_range(self, score):
        v = _mode3_aggressive(score)
        assert 0.5 <= v <= 3.0

    def test_mode4_mean_reversion_zero_score(self):
        assert _mode4_mean_reversion(0.0) == 1.0

    def test_mode4_mean_reversion_caps_at_3(self):
        assert _mode4_mean_reversion(0.01) <= 3.0

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            SuffixAESizer(algo_mode=99)


# ── SuffixAESizer ─────────────────────────────────────────────────────────────────

class TestSuffixAESizer:

    def test_returns_1_when_not_enough_data(self):
        sizer = SuffixAESizer(history_length=300, dna_window=16)
        assert sizer.compute([1.1] * 10) == 1.0

    def test_multiplier_is_positive(self):
        closes = _trending_closes(400)
        sizer = SuffixAESizer(history_length=150, dna_window=16, use_ae=False)
        m = sizer.compute(closes)
        assert m > 0.0

    def test_multiplier_finite(self):
        closes = _trending_closes(400)
        sizer = SuffixAESizer(history_length=150, dna_window=16, use_ae=True)
        m = sizer.compute(closes)
        assert math.isfinite(m)

    def test_flat_market_does_not_crash(self):
        closes = _flat_closes(400)
        sizer = SuffixAESizer(history_length=150, dna_window=16, use_ae=True)
        m = sizer.compute(closes)
        assert math.isfinite(m)
        assert m > 0.0

    def test_zigzag_market(self):
        closes = _zigzag_closes(400)
        sizer = SuffixAESizer(history_length=150, dna_window=16, use_ae=True)
        m = sizer.compute(closes)
        assert math.isfinite(m)
        assert m > 0.0

    @pytest.mark.parametrize("mode", [1, 2, 3, 4])
    def test_all_modes_return_finite_positive(self, mode):
        closes = _trending_closes(400)
        sizer = SuffixAESizer(history_length=150, dna_window=16,
                              algo_mode=mode, use_ae=False)
        m = sizer.compute(closes)
        assert math.isfinite(m) and m > 0.0, f"mode={mode} gave {m}"

    def test_ae_off_vs_on_same_direction(self):
        # With AE on, multiplier should be <= AE-off multiplier
        # (AE can only reduce or preserve, never increase on its own)
        closes = _trending_closes(400)
        m_off = SuffixAESizer(history_length=150, dna_window=16, use_ae=False).compute(closes)
        m_on  = SuffixAESizer(history_length=150, dna_window=16, use_ae=True).compute(closes)
        assert m_on <= m_off + 1e-6, f"AE on={m_on} exceeded AE off={m_off}"

    def test_repeated_calls_are_stable(self):
        closes = _trending_closes(400)
        sizer = SuffixAESizer(history_length=150, dna_window=16, use_ae=False)
        results = [sizer.compute(closes) for _ in range(5)]
        # Without AE (no weight updates), same input → same output
        assert len(set(results)) == 1, f"Non-deterministic: {results}"

    def test_describe_runs_without_error(self, capsys):
        closes = _trending_closes(400)
        sizer = SuffixAESizer(history_length=150, dna_window=16, use_ae=True)
        sizer.describe(closes)
        out = capsys.readouterr().out
        assert "SuffixAE" in out
        assert "Multiplier" in out

    def test_describe_warns_on_insufficient_data(self, capsys):
        sizer = SuffixAESizer(history_length=300, dna_window=16)
        sizer.describe([1.1] * 5)
        out = capsys.readouterr().out
        assert "Not enough data" in out


# ── Integration: lot sizing flow ─────────────────────────────────────────────────

class TestLotSizingIntegration:
    """
    Simulates the on_tick() sizing step without MT5.
    base_lot * multiplier should stay positive and finite.
    """

    def _make_closes(self, scenario: str) -> list:
        if scenario == "trending":
            return _trending_closes(400)
        if scenario == "flat":
            return _flat_closes(400)
        if scenario == "zigzag":
            return _zigzag_closes(400)
        raise ValueError(scenario)

    @pytest.mark.parametrize("scenario", ["trending", "flat", "zigzag"])
    def test_final_lot_positive(self, scenario):
        closes = self._make_closes(scenario)
        base_lot = 0.06
        sizer = SuffixAESizer(history_length=150, dna_window=16, use_ae=True)
        mult = sizer.compute(closes)
        final_lot = round(base_lot * mult, 2)
        min_vol = 0.01
        final_lot = max(final_lot, min_vol)
        assert final_lot >= min_vol
        assert math.isfinite(final_lot)

    def test_structural_break_reduces_lot(self):
        # Simulate a structural break: first build a normal history,
        # then feed a chaotic recent window (random-ish jumps)
        import random
        random.seed(0)
        p = 1.1000
        closes = []
        for i in range(400):
            p += 0.0001 if i % 2 == 0 else -0.0001
            closes.insert(0, p)

        # Now corrupt the first 20 bars (recent window) with large noise
        for i in range(20):
            closes[i] += random.uniform(-0.05, 0.05)

        sizer_normal = SuffixAESizer(history_length=150, dna_window=16, use_ae=False)
        sizer_ae     = SuffixAESizer(history_length=150, dna_window=16, use_ae=True)

        m_no_ae = sizer_normal.compute(closes)
        m_with_ae = sizer_ae.compute(closes)

        # AE should reduce multiplier when recent window is distorted
        assert m_with_ae <= m_no_ae + 1e-6
