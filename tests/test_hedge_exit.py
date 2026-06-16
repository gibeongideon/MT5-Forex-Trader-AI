"""Unit tests for the restart-proof hedge_exit kept-loser close logic."""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _pos(ticket, typ, profit, magic=20260103):
    # mimic the MT5 position fields the helper uses
    return SimpleNamespace(ticket=ticket, type=typ, profit=profit, magic=magic)


class _Bot:
    """Minimal stand-in exposing the unbound helper with a magic."""
    magic = 20260103
    from src.bots.pipeline_bot import PipelineBot
    _hedge_loser_to_close = PipelineBot._hedge_loser_to_close


def test_opposite_pair_older_leg_recovered_returns_older():
    bot = _Bot()
    # ticket 100 (buy, opened first, now +0.5) + ticket 200 (sell, opened later, -1.0)
    pos = [_pos(100, 0, +0.5), _pos(200, 1, -1.0)]
    assert bot._hedge_loser_to_close(pos) == 100  # older leg, recovered → close it


def test_opposite_pair_older_leg_still_losing_returns_none():
    bot = _Bot()
    pos = [_pos(100, 0, -2.0), _pos(200, 1, +1.0)]   # older still in loss
    assert bot._hedge_loser_to_close(pos) is None


def test_breakeven_exactly_zero_closes():
    bot = _Bot()
    pos = [_pos(100, 1, 0.0), _pos(200, 0, -0.3)]    # older at exactly 0 → close (>=0)
    assert bot._hedge_loser_to_close(pos) == 100


def test_single_position_no_hedge_returns_none():
    bot = _Bot()
    assert bot._hedge_loser_to_close([_pos(100, 0, +5.0)]) is None


def test_same_direction_only_returns_none():
    bot = _Bot()
    # two buys (shouldn't happen, but no opposite pair) → no hedge close
    assert bot._hedge_loser_to_close([_pos(100, 0, +1.0), _pos(200, 0, -1.0)]) is None


def test_ignores_other_magic():
    bot = _Bot()
    pos = [_pos(100, 0, +0.5, magic=999), _pos(200, 1, -1.0)]  # buy is another bot's
    assert bot._hedge_loser_to_close(pos) is None  # only one of OUR positions → no pair


def test_older_is_min_ticket_regardless_of_order():
    bot = _Bot()
    pos = [_pos(300, 1, -1.0), _pos(150, 0, +0.2)]   # list order shuffled; 150 is older
    assert bot._hedge_loser_to_close(pos) == 150


# ── recovery-set (persisted, restart-proof, direction-independent) ──────────────
class _RecBot:
    """Stand-in exposing the recovery helpers with a fixed recovery set."""
    from src.bots.pipeline_bot import PipelineBot
    magic = 20260103
    _hedge_loser_to_close = PipelineBot._hedge_loser_to_close
    _recovery_closeable = PipelineBot._recovery_closeable

    def __init__(self, recovery):
        self._recovery_tickets = set(recovery)


def test_lone_recovery_trade_closes_at_breakeven():
    # recovery trade #1, opposite already closed → LONE same-direction, now +0.2
    bot = _RecBot({1})
    pos = [_pos(1, 0, +0.2), _pos(2, 0, -1.0)]   # both BUY (no opposite pair)
    assert bot._recovery_closeable(pos) == {1}    # the pair-fallback would miss this


def test_lone_recovery_still_losing_not_closed():
    bot = _RecBot({1})
    pos = [_pos(1, 0, -0.5), _pos(2, 0, -1.0)]
    assert bot._recovery_closeable(pos) == set()


def test_recovery_closeable_combines_set_and_fallback():
    bot = _RecBot({5})
    # recovery #5 (lone buy, +) AND an opposite pair whose older leg #10 is +
    pos = [_pos(5, 0, +0.1), _pos(10, 0, -1.0), _pos(11, 1, +2.0)]
    # older of opposite pair is #10 but it's losing → fallback None; only #5 closes
    assert bot._recovery_closeable(pos) == {5}


def test_recovery_purges_closed_tickets():
    bot = _RecBot({1, 2, 3})           # 2,3 no longer open
    pos = [_pos(1, 0, +0.5)]
    assert bot._recovery_closeable(pos) == {1}   # only open+profitable returned
