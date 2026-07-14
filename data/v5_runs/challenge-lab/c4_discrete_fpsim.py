"""Campaign 4: the EXACT challenge-bot variant — discrete trade engine
(xau_trend.run_trades, 3xATR trail, conf scale) with champion signal at
risk_frac 0.9%, plus the intraday guard model — FundingPips 2-Step sim.

Guard model (matches the bot's design):
  - daily guard: 15-min cadence flatten at -3.5% from day anchor. Modeled on
    daily bars as: if raw day return <= -7% (gap-through, guard outrun) the
    daily rule is BREACHED; else day loss is capped at -3.8% (guard level
    + slippage allowance) and the day counts as a normal (bad) day.
  - overall halt at -8% is not simulated as a fail-saver (it fires just
    before the firm's -10%); maxloss fails are counted at -10% as before.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from challenge_lab import *  # noqa

LABX = "/home/rock/Desktop/2026_Projects/Trader36/MT5/data/v5_runs/xau-sharpe1-lab"
sys.path.insert(0, LABX)
import src.v5.xau_trend as xt  # noqa
from src.v5.xau_dual_signals import champion_signal  # noqa

h4 = load_xau_h4()
fc = champion_signal(h4["close"])
xt.xau_signal = lambda close: fc.reindex(close.index).fillna(0.0)

CS = {"low": 0.5, "med": 1.0, "high": 1.5}
res = xt.run_trades(h4, equity0=100_000.0, exit_mode="trail",
                    flip_mode="confidence",
                    params={"sl_atr": 3.0, "trail_atr": 3.0,
                            "conf_risk_scale": CS, "risk_frac": 0.009})
eq = res["equity"].dropna().loc["2016-01-01":]
d = eq.resample("D").last().pct_change(fill_method=None).dropna()
d = d[d.index.dayofweek < 5]
print("DISCRETE stream @risk0.9%:", {k: v for k, v in stats(d).items() if k != 'tag'})

r = d.values

def fp_sim_guard(r, nsim=4000, block=20, maxd=2520, p1=0.08, p2=0.05,
                 dayloss=0.05, maxloss=0.10, guard_cap=0.038,
                 gap_through=0.07, use_guard=True, seed=7):
    rng = np.random.default_rng(seed)
    n = len(r)
    passed = fail_day = fail_dd = 0
    days_list = []
    for _ in range(nsim):
        idx = []
        while len(idx) < maxd:
            s = rng.integers(0, n)
            L = min(rng.geometric(1 / block), maxd - len(idx))
            idx.extend([(s + j) % n for j in range(L)])
        x = r[np.array(idx)]
        eqv, base, tgt, day, ok, ph2, isday = 1.0, 1.0, 1 + p1, 0, None, False, False
        for dd_ in x:
            day += 1
            if use_guard:
                if dd_ <= -gap_through:      # gap outruns the guard -> breach
                    ok, isday = False, True
                    break
                dd_ = max(dd_, -guard_cap)   # guard caps the day's loss
            else:
                if dd_ < -dayloss:
                    ok, isday = False, True
                    break
            eqv *= (1 + dd_)
            if eqv < base * (1 - maxloss):
                ok, isday = False, False
                break
            if eqv >= base * tgt:
                if not ph2:
                    ph2, base, tgt = True, eqv, 1 + p2
                else:
                    ok = True
                    break
        if ok is True:
            passed += 1
            days_list.append(day)
        elif ok is False:
            fail_day += isday
            fail_dd += not isday
    med = np.median(days_list) if days_list else -1
    q25 = np.percentile(days_list, 25) if days_list else -1
    q75 = np.percentile(days_list, 75) if days_list else -1
    return dict(passpct=round(passed / nsim * 100, 1),
                fail_day=round(fail_day / nsim * 100, 1),
                fail_dd=round(fail_dd / nsim * 100, 1),
                p25_mo=round(q25 / 21, 1), med_mo=round(med / 21, 1),
                q75_mo=round(q75 / 21, 1))

print("\n=== EXACT BOT, FundingPips 2-Step, WITH intraday guard ===")
for k in (0.8, 1.0, 1.2):
    s = fp_sim_guard(k * r)
    print(f"risk {0.9*k:.2f}%: pass {s['passpct']}%  failDay {s['fail_day']}%  "
          f"failDD {s['fail_dd']}%  time p25/med/p75 = "
          f"{s['p25_mo']}/{s['med_mo']}/{s['q75_mo']} mo")

print("\n--- no guard (for contrast) ---")
s = fp_sim_guard(r, use_guard=False)
print(f"risk 0.90%: pass {s['passpct']}%  failDay {s['fail_day']}%  "
      f"failDD {s['fail_dd']}%  med {s['med_mo']} mo")

print("\n--- stress on the guarded bot @0.9% ---")
w = d.loc[:"2020-12-31"].values
s = fp_sim_guard(w); print(f"weak-years  : pass {s['passpct']}%  med {s['med_mo']} mo")
hc = r - r.mean() * 0.25
s = fp_sim_guard(hc); print(f"edge -25%   : pass {s['passpct']}%  med {s['med_mo']} mo")
s = fp_sim_guard(r, gap_through=0.05)
print(f"gapthru @5% : pass {s['passpct']}%  med {s['med_mo']} mo")
