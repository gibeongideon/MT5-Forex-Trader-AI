"""Pure guard/state logic for the FundingPips challenge bot (testable).

State dict keys (persisted as JSON by the executor):
  initial_balance   float  set once on first pass
  phase             int    1 or 2
  phase_start       float  balance at phase start
  phase_target_frac float  0.08 (P1) / 0.05 (P2)
  day_anchor        float  max(balance, equity) at first pass of the day
  day_anchor_date   str    'YYYY-MM-DD' (platform time, UTC+3)
  day_locked        bool   daily guard tripped today -> no trading till reset
  halted            bool   overall guard tripped -> permanent flat
  phase_complete    bool   target realized -> stop trading, await promotion

All checks are conservative: they use EQUITY (includes floating), like the
firm's rules. Fractions are of the anchor/initial, not of current equity.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

UTC3 = timezone(timedelta(hours=3))

# Reset timezone for the daily anchor. FundingPips = UTC+3; FTMO = 00:00
# CE(S)T (Europe/Prague, DST-aware). Executor sets RESET_TZ from config.
RESET_TZ = UTC3


def set_reset_tz(name: str | None) -> None:
    """Set the daily-reset timezone by name: 'UTC+3'/'FP' (FundingPips) or
    'CET'/'CEST'/'Europe/Prague'/'FTMO' (FTMO, DST-aware if zoneinfo present)."""
    global RESET_TZ
    if not name:
        return
    key = name.strip().lower()
    if key in ("utc+3", "fp", "fundingpips"):
        RESET_TZ = UTC3
        return
    if key in ("cet", "cest", "ftmo", "europe/prague"):
        try:
            from zoneinfo import ZoneInfo
            RESET_TZ = ZoneInfo("Europe/Prague")
        except Exception:
            RESET_TZ = timezone(timedelta(hours=2))  # CEST fixed fallback
        return
    # explicit "UTC+N"
    if key.startswith("utc") and (key[3:] or "").lstrip("+-").isdigit():
        RESET_TZ = timezone(timedelta(hours=int(key[3:])))

DAILY_GUARD_FRAC = 0.035   # flatten at -3.5% from day anchor (firm: -5%)
OVERALL_HALT_FRAC = 0.08   # permanent halt at -8% from initial (firm: -10%)
PHASE_TARGETS = {1: 0.08, 2: 0.05}


def platform_date(now_utc: datetime | None = None) -> str:
    now = now_utc or datetime.now(timezone.utc)
    return now.astimezone(RESET_TZ).strftime("%Y-%m-%d")


def init_state(balance: float, equity: float,
               now_utc: datetime | None = None) -> dict:
    return dict(initial_balance=float(balance), phase=1,
                phase_start=float(balance),
                phase_target_frac=PHASE_TARGETS[1],
                day_anchor=float(max(balance, equity)),
                day_anchor_date=platform_date(now_utc),
                day_locked=False, halted=False, phase_complete=False)


def roll_day(state: dict, balance: float, equity: float,
             now_utc: datetime | None = None) -> dict:
    """At the first pass of a new platform day: reset anchor + day lock."""
    today = platform_date(now_utc)
    if state["day_anchor_date"] != today:
        state = dict(state)
        state["day_anchor"] = float(max(balance, equity))
        state["day_anchor_date"] = today
        state["day_locked"] = False
    return state


def daily_guard_hit(state: dict, equity: float,
                    frac: float = DAILY_GUARD_FRAC) -> bool:
    return equity <= state["day_anchor"] * (1.0 - frac)


def overall_halt_hit(state: dict, equity: float,
                     frac: float = OVERALL_HALT_FRAC) -> bool:
    return equity <= state["initial_balance"] * (1.0 - frac)


def phase_target_hit(state: dict, equity: float) -> bool:
    return equity >= state["phase_start"] * (1.0 + state["phase_target_frac"])


def phase_target_realized(state: dict, balance: float) -> bool:
    """Realized (closed) profit at/above target — what the firm grades."""
    return balance >= state["phase_start"] * (1.0 + state["phase_target_frac"])


def advance_phase(state: dict, balance: float) -> dict:
    """Manual promotion: call when the firm issues Phase-2 credentials."""
    state = dict(state)
    state["phase"] = 2
    state["phase_start"] = float(balance)
    state["phase_target_frac"] = PHASE_TARGETS[2]
    state["phase_complete"] = False
    return state


def decide(state: dict, balance: float, equity: float,
           now_utc: datetime | None = None) -> tuple[dict, str]:
    """One guard evaluation. Returns (new_state, action):
      action in {'halt', 'day_lock', 'locked', 'realize_target',
                 'complete', 'trade'}.
    Executor mapping: halt/day_lock/locked/complete -> ensure FLAT, no entries;
    realize_target -> close open position to bank the target, then re-check;
    trade -> normal reconcile pass.
    """
    state = roll_day(state, balance, equity, now_utc)
    if state["halted"]:
        return state, "halt"
    if overall_halt_hit(state, equity):
        state = dict(state)
        state["halted"] = True
        return state, "halt"
    if state["phase_complete"]:
        return state, "complete"
    if phase_target_realized(state, balance) and \
            not _has_floating(balance, equity):
        state = dict(state)
        state["phase_complete"] = True
        return state, "complete"
    if state["day_locked"]:
        return state, "locked"
    if daily_guard_hit(state, equity):
        state = dict(state)
        state["day_locked"] = True
        return state, "day_lock"
    if phase_target_hit(state, equity):
        return state, "realize_target"
    return state, "trade"


def _has_floating(balance: float, equity: float, tol: float = 1e-6) -> bool:
    return abs(equity - balance) > tol * max(abs(balance), 1.0)
