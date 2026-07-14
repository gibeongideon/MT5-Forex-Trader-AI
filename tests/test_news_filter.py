"""News-filter logic tests (no network): window math, adaptive majors,
currency/impact filtering, fail-open, and plan filtering with the
close-in-profit rule."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.v5.news_filter import DEFAULTS, apply_to_plan, check_events

CFG = {**DEFAULTS, "enabled": True}
T = datetime(2026, 7, 17, 12, 30, tzinfo=timezone.utc)


def ev(title="Retail Sales m/m", country="USD", impact="High", when=T):
    return dict(title=title, country=country, impact=impact,
                date=when.strftime("%Y-%m-%dT%H:%M:%S+00:00"))


def test_blocks_inside_standard_window():
    for offset in (-30, -1, 0, 15, 30):
        v = check_events([ev(when=T - timedelta(minutes=offset))], T, CFG)
        assert v["blocked"], offset


def test_clear_outside_standard_window():
    for offset in (-31, 31, 120):
        v = check_events([ev(when=T + timedelta(minutes=offset))], T, CFG)
        assert not v["blocked"], offset


def test_major_events_get_wider_window():
    nfp = ev(title="Non-Farm Employment Change", when=T + timedelta(minutes=45))
    assert check_events([nfp], T, CFG)["blocked"]          # 45 < 60 before
    plain = ev(when=T + timedelta(minutes=45))
    assert not check_events([plain], T, CFG)["blocked"]    # 45 > 30 before
    fomc = ev(title="FOMC Statement", when=T - timedelta(minutes=40))
    assert check_events([fomc], T, CFG)["blocked"]         # 40 < 45 after


def test_ignores_low_impact_and_other_currencies():
    assert not check_events([ev(impact="Medium")], T, CFG)["blocked"]
    assert not check_events([ev(country="EUR")], T, CFG)["blocked"]


def test_bad_dates_fail_open():
    e = ev()
    e["date"] = "not-a-date"
    assert not check_events([e], T, CFG)["blocked"]
    assert not check_events([], T, CFG)["blocked"]


def _pos(profit):
    return SimpleNamespace(ticket=111, profit=profit)


BLOCKED = dict(blocked=True, event="CPI m/m")
CLEAR = dict(blocked=False)


def test_plan_untouched_when_clear():
    plan = [("open_market", {"dir": 1})]
    out, nblk, added = apply_to_plan(plan, _pos(5.0), CLEAR, True)
    assert out == plan and nblk == 0 and not added


def test_blocks_new_entries_keeps_management():
    plan = [("modify_sl", {"sl": 4000.0}), ("open_market", {"dir": 1})]
    out, nblk, added = apply_to_plan(plan, None, BLOCKED, True)
    assert nblk == 1 and not added
    assert [a for a, _ in out] == ["modify_sl"]


def test_closes_profitable_position_in_window():
    out, nblk, added = apply_to_plan([], _pos(12.0), BLOCKED, True)
    assert added and out[0][0] == "close"
    assert out[0][1]["why"] == "news_profit_close"


def test_losing_position_kept_open():
    out, nblk, added = apply_to_plan([], _pos(-3.0), BLOCKED, True)
    assert not added and out == []


def test_no_double_close():
    plan = [("close", {"ticket": 111})]
    out, nblk, added = apply_to_plan(plan, _pos(9.0), BLOCKED, True)
    assert not added and [a for a, _ in out] == ["close"]


def test_close_in_profit_configurable_off():
    out, nblk, added = apply_to_plan([], _pos(12.0), BLOCKED, False)
    assert not added and out == []
