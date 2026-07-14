"""Campaign 5: variations on the long-only max-combination + CI/stress
for the leaders. LO-max-ew-bko: eval 0.951 / full 0.879."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from xau_lab import *  # noqa
from src.cta.bootstrap import block_bootstrap_sharpe  # noqa

h4 = load_h4()
D = 6
MID = tuple((f * D, s * D) for f, s in ((16, 64), (32, 128), (64, 256)))

base = ewmac_fc(h4["close"], MID)
bko_f = breakout_fc(h4["close"], [d * D for d in (10, 20, 40)])
bko_vf = breakout_fc(h4["close"], [d * D for d in (5, 10, 20)])
ts = tsmom_fc(h4["close"], [d * D for d in (21, 63, 126, 252)])

L = lambda s: s.clip(lower=0.0)

cands = {
    "LO-max3-ew-bko-ts": np.maximum(np.maximum(L(base), L(bko_f)), L(ts)),
    "LO-max-ew-bkovf": np.maximum(L(base), L(bko_vf)),
    "LO-max-ew-bko-rest0.15": (np.maximum(L(base), L(bko_f)) + 0.15).clip(0, 2),
    "LO-max-ew-bko-rest0.25": (np.maximum(L(base), L(bko_f)) + 0.25).clip(0, 2),
    "LO-avg-maxewbko-ts": 0.7 * np.maximum(L(base), L(bko_f)) + 0.3 * L(ts),
    "LO-bko-veryfast": L(bko_vf),
}
for nm, fc in cands.items():
    m = run(h4, fc, ann=ANN_H4)
    log_result(nm, {}, m)

# ---- stress battery + bootstrap CI on the leaders
def stress(nm, fc):
    for tag, kw in [("costx2", dict(spread_mult=2.0)),
                    ("delay2", dict(delay=2)),
                    ("buf0.2", dict(buffer_frac=0.2))]:
        m = run(h4, fc, ann=ANN_H4, **kw)
        log_result(f"STRESS-{nm}-{tag}", {"stress": tag}, m)
    # subwindow sharpes + CI (daily resample of net)
    close = h4["close"]
    ret = close.pct_change()
    vol = ret.ewm(halflife=42, min_periods=20).std() * np.sqrt(ANN_H4)
    pos = (fc * (0.10 / vol)).clip(-8, 8).shift(1).fillna(0.0)
    cost = pos.diff().abs().fillna(0) * ((h4["spread_px"] / 2 + SLIP_USD) / close)
    net = (pos * ret - cost).fillna(0.0)
    for tag, sl in [("2017+", slice("2017-01-01", None)),
                    ("2021+", slice("2021-01-01", None)),
                    ("2015-2020", slice(None, "2020-12-31"))]:
        d = net.loc[sl].resample("D").sum()
        d = d[d != 0.0]
        sh = d.mean() / d.std() * np.sqrt(252)
        lo, hi = block_bootstrap_sharpe(d.values)
        print(f"CI     {nm:28s} {tag:10s} SR {sh:+.3f}  CI95 [{lo:+.2f}, {hi:+.2f}]")

stress("max-ew-bko", np.maximum(L(base), L(bko_f)))
stress("bko-fast", L(bko_f))
