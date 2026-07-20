"""FTMO 2-Step challenge — daily email report: where we are in the challenge
+ compliance %. Reads the live FTMO account (via the local MT5 bridge), the
executor state, and the guard log; emails via SMTP (creds in .env.mail, else
prints). Mirrors the VPS basket report but on FTMO rules / CE(S)T.

    conda run -n envmt5 python scripts/ftmo_daily_report.py
"""
from __future__ import annotations
import sys, json, csv, smtplib, ssl
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.core.mt5_connector import get_mt5   # noqa: E402

CFG = json.loads((ROOT / "configs" / "v5_ftmo_challenge.json").read_text())
MAGIC = CFG["magic"]
STATE = ROOT / "data" / "v5_runs" / "ftmo_challenge_state.json"
GLOG = ROOT / "data" / "v5_runs" / "ftmo_challenge_guard_log.csv"
LLOG = ROOT / "data" / "v5_runs" / "ftmo_challenge_live_log.csv"
DAILY_LIMIT, MAX_LIMIT = 0.05, 0.10   # FTMO firm limits (2-Step)
PHASE_TARGETS = {int(k): float(v) for k, v in CFG["guards"]["phase_targets"].items()}
MIN_DAYS = int(CFG.get("min_trading_days", 4))

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("Europe/Prague")
except Exception:
    from datetime import timedelta
    TZ = timezone(timedelta(hours=2))


def _bars_of_progress(pct, width=20):
    n = max(0, min(width, int(round(pct / 100 * width))))
    return "█" * n + "·" * (width - n)


def worst_day_dd_today(anchor):
    """Most negative intraday day-dd logged today (CET)."""
    today = datetime.now(TZ).strftime("%Y-%m-%d"); worst = 0.0
    for f in (GLOG, LLOG):
        if not f.exists():
            continue
        for r in csv.DictReader(open(f)):
            try:
                d = datetime.strptime(r.get("time_utc", ""), "%Y-%m-%d %H:%M:%S") \
                    .replace(tzinfo=timezone.utc).astimezone(TZ).strftime("%Y-%m-%d")
            except Exception:
                continue
            if d == today:
                try:
                    worst = min(worst, float(r.get("day_dd_pct", 0)))
                except Exception:
                    pass
    return worst


def trading_days(m):
    """FTMO trading day = a day with >=1 position opened. Count distinct CE(S)T
    dates among our entry deals this challenge (from MT5 deal history)."""
    # NOTE: MT5 history uses SERVER time (this broker is GMT+3), so a UTC "now"
    # end-bound silently drops today's deals. Pad the window generously.
    try:
        from datetime import timedelta
        deals = m.history_deals_get(datetime(2020, 1, 1, tzinfo=timezone.utc),
                                    datetime.now(timezone.utc) + timedelta(days=3))
    except Exception:
        return None
    if not deals:
        return 0
    days = set()
    for d in deals:
        if getattr(d, "magic", 0) != MAGIC:
            continue
        if getattr(d, "entry", None) == 0:   # DEAL_ENTRY_IN (opening)
            ts = datetime.fromtimestamp(d.time, timezone.utc).astimezone(TZ)
            days.add(ts.strftime("%Y-%m-%d"))
    return len(days)


def build():
    m = get_mt5("localhost", 18812)
    if not m.initialize():
        return None, "MT5 not connected (check terminal + master password)"
    a = m.account_info()
    ps = [p for p in (m.positions_get() or []) if p.magic == MAGIC]
    floating = sum(p.profit for p in ps)
    tdays = trading_days(m)
    m.shutdown()

    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    init = st.get("initial_balance", a.balance)
    pstart = st.get("phase_start", init)
    anchor = st.get("day_anchor", a.balance)
    phase = st.get("phase", 1)
    target = PHASE_TARGETS.get(phase, 0.10)

    day_pl = a.equity - anchor
    total_pl = a.equity - init
    day_dd = min(0.0, a.equity / anchor - 1) * 100
    worst_day = min(day_dd, worst_day_dd_today(anchor))
    total_dd = min(0.0, a.equity / init - 1) * 100
    prog = (a.equity / pstart - 1) / target * 100      # % of phase target hit

    daily_used = abs(min(0.0, worst_day / 100)) / DAILY_LIMIT * 100
    max_used = abs(min(0.0, total_dd / 100)) / MAX_LIMIT * 100
    compliance = 100 - max(daily_used, max_used)
    viol = ("NONE" if (worst_day / 100 > -DAILY_LIMIT and total_dd / 100 > -MAX_LIMIT)
            else "⚠ BREACH")
    days_ok = "OK" if (tdays is None or tdays >= MIN_DAYS) else f"need {MIN_DAYS - tdays} more"
    d = datetime.now(TZ)

    body = f"""FTMO 2-Step Challenge — Daily Summary
{d:%A %d %b %Y}  (platform day, CE(S)T)   account {a.login} @ {a.server}

BALANCE / EQUITY   {a.balance:,.0f} / {a.equity:,.0f} {a.currency}
TODAY'S GAIN       {day_pl:+,.0f} {a.currency}   ({(a.equity/anchor-1)*100:+.2f}%)
TOTAL P&L          {total_pl:+,.0f} {a.currency}   ({(a.equity/init-1)*100:+.2f}% from start)
OPEN POSITIONS     {len(ps)}   floating {floating:+,.0f} {a.currency}

PHASE {phase}  ->  target +{target*100:.0f}%
  PROGRESS   {prog:5.1f}% of target   [{_bars_of_progress(max(0,prog))}]
  TRADING DAYS  {tdays if tdays is not None else '?'}/{MIN_DAYS}  ({days_ok})

COMPLIANCE         {compliance:5.1f}%   (violations today: {viol})
  Daily-loss  limit -{DAILY_LIMIT*100:.0f}%   | worst today {worst_day:+.2f}%  | used {daily_used:4.0f}%  | headroom {100-daily_used:4.0f}%
  Max-loss    limit -{MAX_LIMIT*100:.0f}%  | current    {total_dd:+.2f}%  | used {max_used:4.0f}%  | headroom {100-max_used:4.0f}%

Book: XAU champion + BTC + NDX (equal 1/3, long-only, 7% vol). Guards: real-time
monitor (flatten at -3.5% day / -8% total), reconcile hourly, daily reset 00:00 CET.
"""
    subj = (f"[FTMO] {d:%d %b} — eq {a.equity:,.0f} {a.currency}, "
            f"P{phase} {prog:.0f}% to target, compliance {compliance:.0f}%")
    return subj, body


def send(subj, body):
    e = {}
    mf = ROOT / ".env.mail"
    if mf.exists():
        for ln in mf.read_text().splitlines():
            if "=" in ln and not ln.strip().startswith("#"):
                k, v = ln.split("=", 1); e[k.strip()] = v.strip()
    host, user, pw = e.get("SMTP_HOST"), e.get("SMTP_USER"), e.get("SMTP_PASS")
    to = e.get("REPORT_TO", "kipngenol@gmail.com"); port = int(e.get("SMTP_PORT", "587"))
    if not (host and user and pw):
        print("NO MAIL CREDS (.env.mail) — report below:\n\n" + body); return
    msg = MIMEText(body); msg["Subject"] = subj; msg["From"] = user; msg["To"] = to
    with smtplib.SMTP(host, port, timeout=30) as s:
        s.starttls(context=ssl.create_default_context()); s.login(user, pw)
        s.send_message(msg)
    print(f"emailed {to}: {subj}")


if __name__ == "__main__":
    subj, body = build()
    if subj is None:
        print("report failed:", body); sys.exit(1)
    send(subj, body)
