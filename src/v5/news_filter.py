"""High-impact news filter for the XAU bots (config-driven, per-bot).

Policy (see CHALLENGEBOT.MD / user spec 2026-07-14):
  * inside a high-impact window for the watched currencies: block NEW
    entries; keep managing existing positions (trail/SL/close actions);
    optionally close an existing position IF IT IS IN PROFIT.
  * outside windows: no effect.

Source: ForexFactory weekly calendar JSON (free, no key):
    https://nfs.faireconomy.media/ff_calendar_thisweek.json
Fields used: title, country (currency code), date (ISO-8601 with offset),
impact ("High"/"Medium"/...). Fetched at most every `refresh_minutes`,
cached to disk so restarts / offline passes reuse the last snapshot.

Fail-open by design: if the feed and the cache are BOTH unavailable the
filter reports "not blocked" (a missing calendar must not silently stop a
trend bot for days) — but callers get `stale=True` to log loudly.

The MetaTrader5 Python API exposes no economic calendar, hence the
external feed. MQL5's native CalendarValueHistory is not reachable from
the bridge.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

DEFAULTS = dict(
    enabled=False,
    currencies=["USD"],
    impact="High",
    before_min=30,
    after_min=30,
    major_before_min=60,
    major_after_min=45,
    close_in_profit=True,
    refresh_minutes=60,
    cache_file="data/news_cache.json",
)

MAJOR_KEYWORDS = (
    "non-farm", "nonfarm", "non farm",
    "fomc", "federal funds rate", "interest rate decision",
    "cpi", "consumer price index",
    "pce price index", "core pce",
    "gdp",
    "unemployment rate",
)


def _is_major(title: str) -> bool:
    t = title.lower()
    return any(k in t for k in MAJOR_KEYWORDS)


def _parse_time(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def check_events(events: list[dict], now_utc: datetime, cfg: dict) -> dict:
    """Pure window logic (unit-testable). Returns dict(blocked, event,
    event_time, window)."""
    for ev in events:
        if str(ev.get("impact", "")).lower() != str(cfg["impact"]).lower():
            continue
        if str(ev.get("country", "")).upper() not in \
                [c.upper() for c in cfg["currencies"]]:
            continue
        t = _parse_time(ev.get("date", ""))
        if t is None:
            continue
        title = str(ev.get("title", ""))
        if _is_major(title):
            before, after = cfg["major_before_min"], cfg["major_after_min"]
        else:
            before, after = cfg["before_min"], cfg["after_min"]
        if t - timedelta(minutes=before) <= now_utc \
                <= t + timedelta(minutes=after):
            return dict(blocked=True, event=title,
                        event_time=t.strftime("%F %T UTC"),
                        window=f"-{before}/+{after}min")
    return dict(blocked=False, event=None, event_time=None, window=None)


class NewsFilter:
    def __init__(self, cfg: dict | None = None, root: Path | str = "."):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.cache = Path(root) / self.cfg["cache_file"]

    # ---------------------------------------------------------- fetching
    def _load_cache(self) -> tuple[list[dict], float]:
        try:
            payload = json.loads(self.cache.read_text())
            return payload["events"], float(payload["fetched_at"])
        except Exception:  # noqa: BLE001 — any cache problem = no cache
            return [], 0.0

    def _fetch(self) -> list[dict]:
        import requests
        r = requests.get(FEED_URL, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0 (xau-bot)"})
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list):
            raise RuntimeError(f"unexpected calendar payload: {type(data)}")
        return data

    def events(self) -> tuple[list[dict], bool]:
        """Cached events. Returns (events, stale). stale=True when the feed
        could not be refreshed and the cache is old (> 2x refresh) or empty."""
        events, fetched_at = self._load_cache()
        age_min = (time.time() - fetched_at) / 60.0
        if age_min >= self.cfg["refresh_minutes"]:
            try:
                events = self._fetch()
                self.cache.parent.mkdir(parents=True, exist_ok=True)
                self.cache.write_text(json.dumps(
                    dict(fetched_at=time.time(), events=events)))
                age_min = 0.0
            except Exception as exc:  # noqa: BLE001 — keep stale cache
                print(f"  ! news feed refresh failed ({exc}) — "
                      f"using cache ({age_min:.0f} min old, "
                      f"{len(events)} events)")
        stale = (not events) or age_min > 2 * self.cfg["refresh_minutes"]
        return events, stale

    # ---------------------------------------------------------- checking
    def check(self, now_utc: datetime | None = None) -> dict:
        """Main entry: dict(blocked, event, event_time, window, stale).
        Disabled or unavailable calendar -> blocked=False (fail-open)."""
        if not self.cfg["enabled"]:
            return dict(blocked=False, event=None, event_time=None,
                        window=None, stale=False)
        now = now_utc or datetime.now(timezone.utc)
        events, stale = self.events()
        out = check_events(events, now, self.cfg)
        out["stale"] = stale
        return out


def apply_to_plan(actions: list, held, verdict: dict, close_in_profit: bool):
    """Filter a reconcile plan for a news window.

    - drops open_market actions (no new entries);
    - keeps close / modify_sl / cancel (management continues);
    - if a position is held and in profit and close_in_profit: prepends a
      close action tagged news_profit_close (unless a close is already
      planned).
    Returns (actions, blocked_entries: int, profit_close_added: bool).
    """
    if not verdict.get("blocked"):
        return actions, 0, False
    kept = [(act, a) for act, a in actions if act != "open_market"]
    blocked = len(actions) - len(kept)
    added = False
    if close_in_profit and held is not None \
            and float(getattr(held, "profit", 0.0)) > 0.0 \
            and not any(act == "close" for act, _ in kept):
        kept.insert(0, ("close", dict(position=held, ticket=held.ticket,
                                      why="news_profit_close")))
        added = True
    return kept, blocked, added
