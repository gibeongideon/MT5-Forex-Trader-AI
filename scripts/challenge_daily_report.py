"""Config-driven prop-challenge daily report: where we are + compliance %.

Works for ANY challenge config (FundingPips Flex, FTMO 2-Step, ...) — the firm
name, phase targets, daily/max loss limits and reset timezone all come from the
config + engine model, so there is one script instead of one per firm.

    conda run -n envmt5 python scripts/challenge_daily_report.py \
        --config configs/v5_fp_flex_10k.json \
        --state  data/v5_runs/fp10k_state.json \
        --port   18812
"""
from __future__ import annotations
import argparse, csv, json, smtplib, ssl, sys
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.core.mt5_connector import get_mt5           # noqa: E402
from v5_basket_challenge import MODELS               # noqa: E402


def tz_from(name: str):
    key = (name or "").strip().lower()
    if key in ("cet", "cest", "ftmo", "europe/prague"):
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo("Europe/Prague")
        except Exception:
            return timezone(timedelta(hours=2))
    if key.startswith("utc") and key[3:].lstrip("+-").isdigit():
        return timezone(timedelta(hours=int(key[3:])))
    return timezone(timedelta(hours=3))              # FundingPips default


def bar(pct, width=20):
    n = max(0, min(width, int(round(pct / 100 * width))))
    return "█" * n + "·" * (width - n)


def worst_day_dd(logs, tz):
    today = datetime.now(tz).strftime("%Y-%m-%d")
    worst = 0.0
    for f in logs:
        if not f.exists():
            continue
        for r in csv.DictReader(open(f)):
            try:
                d = datetime.strptime(r.get("time_utc", ""), "%Y-%m-%d %H:%M:%S") \
                    .replace(tzinfo=timezone.utc).astimezone(tz).strftime("%Y-%m-%d")
            except Exception:
                continue
            if d == today:
                try:
                    worst = min(worst, float(r.get("day_dd_pct", 0)))
                except Exception:
                    pass
    return worst


def trading_days(m, magic, tz):
    """Days with >=1 position OPENED. MT5 history uses SERVER time, so pad the
    end bound generously or today's deals are silently dropped."""
    try:
        deals = m.history_deals_get(datetime(2020, 1, 1, tzinfo=timezone.utc),
                                    datetime.now(timezone.utc) + timedelta(days=3))
    except Exception:
        return None
    if not deals:
        return 0
    days = set()
    for d in deals:
        if getattr(d, "magic", 0) != magic or getattr(d, "entry", None) != 0:
            continue
        days.add(datetime.fromtimestamp(d.time, timezone.utc).astimezone(tz).strftime("%Y-%m-%d"))
    return len(days)


def build(cfg_path: Path, state_path: Path, port: int):
    cfg = json.loads(cfg_path.read_text())
    model = MODELS[cfg["model"]]
    firm = cfg.get("firm", "Challenge")
    magic = cfg["magic"]
    tz = tz_from(cfg.get("reset_tz"))
    daily_lim, max_lim = model["daily"], model["maxloss"]
    targets = {int(k): float(v) for k, v in cfg["guards"]["phase_targets"].items()}
    min_days = int(cfg.get("min_trading_days", 0))

    m = get_mt5("localhost", port)
    if not m.initialize():
        return None, f"MT5 not connected on bridge {port}"
    a = m.account_info()
    ps = [p for p in (m.positions_get() or []) if p.magic == magic]
    floating = sum(p.profit for p in ps)
    tdays = trading_days(m, magic, tz)
    m.shutdown()

    st = json.loads(state_path.read_text()) if state_path.exists() else {}
    init = st.get("initial_balance", a.balance)
    pstart = st.get("phase_start", init)
    anchor = st.get("day_anchor", a.balance)
    phase = st.get("phase", 1)
    target = targets.get(phase, 0.10)

    day_pl, total_pl = a.equity - anchor, a.equity - init
    day_dd = min(0.0, a.equity / anchor - 1) * 100
    worst = min(day_dd, worst_day_dd(
        [state_path.parent / f"{state_path.stem.replace('_state','')}_guard_log.csv",
         state_path.parent / f"{state_path.stem.replace('_state','')}_live_log.csv"], tz))
    total_dd = min(0.0, a.equity / init - 1) * 100
    prog = (a.equity / pstart - 1) / target * 100

    daily_used = abs(min(0.0, worst / 100)) / daily_lim * 100
    max_used = abs(min(0.0, total_dd / 100)) / max_lim * 100
    compliance = 100 - max(daily_used, max_used)
    viol = ("NONE" if (worst / 100 > -daily_lim and total_dd / 100 > -max_lim)
            else "⚠ BREACH")
    days_txt = (f"{tdays}/{min_days}" if tdays is not None else "?") + \
               ("" if (tdays is None or not min_days or tdays >= min_days)
                else f"  (need {min_days - tdays} more)")
    d = datetime.now(tz)
    pos_lines = "\n".join(
        f"    {p.symbol:14s} {'BUY ' if p.type == 0 else 'SELL'} {p.volume:6.2f} "
        f"@ {p.price_open:<10.2f} P/L {p.profit:+8.2f}" for p in ps) or "    (flat)"

    body = f"""{firm} {cfg['model'].upper()} Challenge — Daily Summary
{d:%A %d %b %Y}  (platform day)   account {a.login} @ {a.server}

BALANCE / EQUITY   {a.balance:,.2f} / {a.equity:,.2f} {a.currency}
TODAY'S GAIN       {day_pl:+,.2f} {a.currency}   ({(a.equity/anchor-1)*100:+.2f}%)
TOTAL P&L          {total_pl:+,.2f} {a.currency}   ({(a.equity/init-1)*100:+.2f}% from start)
OPEN POSITIONS     {len(ps)}   floating {floating:+,.2f} {a.currency}
{pos_lines}

PHASE {phase}  ->  target +{target*100:.0f}%
  PROGRESS   {prog:6.1f}% of target   [{bar(max(0, prog))}]
  TRADING DAYS  {days_txt}

COMPLIANCE         {compliance:6.1f}%   (violations today: {viol})
  Daily-loss  limit -{daily_lim*100:.0f}%   | worst today {worst:+.2f}%  | used {daily_used:4.0f}%  | headroom {100-daily_used:4.0f}%
  Max-loss    limit -{max_lim*100:.0f}%  | current    {total_dd:+.2f}%  | used {max_used:4.0f}%  | headroom {100-max_used:4.0f}%

Guards: flatten at -{cfg['guards']['daily_guard_frac']*100:.1f}% daily / halt -{cfg['guards']['overall_halt_frac']*100:.0f}% total.
Book: {', '.join(sum(cfg.get('classes', {}).values(), []))}   magic {magic}
"""
    subj = (f"[{firm} {cfg['model'].upper()}] {d:%d %b} — eq {a.equity:,.0f} {a.currency}, "
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
        s.starttls(context=ssl.create_default_context()); s.login(user, pw); s.send_message(msg)
    print(f"emailed {to}: {subj}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--port", type=int, default=18812)
    args = ap.parse_args()
    subj, body = build(Path(args.config), Path(args.state), args.port)
    if subj is None:
        print("report failed:", body); sys.exit(1)
    send(subj, body)
