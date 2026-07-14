"""Campaign 5: H1 trend stream (timeframe diversification) + bar-of-day
long-drift test on H4. H1 data 2015-01 .. 2026-06."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from xau_lab import *  # noqa

h4 = load_h4()
D = 6

def load_h1():
    df = pd.read_csv(f"{ROOT}/data/XAUUSD_H1_long.csv",
                     parse_dates=["time"], index_col="time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["spread_px"] = np.maximum(df["spread"], df["spread"].median()) * 0.1
    return df

h1 = load_h1()
ANN_H1 = 252 * 24

# --- H1 EWMAC long-flat (daily-equivalent mid speeds x24)
MID_H1 = tuple((f * 24, s * 24) for f, s in ((16, 64), (32, 128), (64, 256)))
FASTER_H1 = tuple((f * 24, s * 24) for f, s in ((4, 16), (8, 32), (16, 64)))
for name, spd in (("mid", MID_H1), ("faster", FASTER_H1)):
    fc = ewmac_fc(h1["close"], spd)
    lf = fc.where(fc > 0, 0.0)
    m = run(h1, lf, ann=ANN_H1, buffer_frac=0.3)
    log_result(f"h1-lf-ewmac-{name}-buf0.3", {}, m)

# --- bar-of-day long drift on H4 (which 4h bars carry gold's drift?)
ret_h4 = h4["close"].pct_change()
stats = ret_h4.groupby(h4.index.hour).agg(["mean", "std", "count"])
stats["t"] = stats["mean"] / (stats["std"] / np.sqrt(stats["count"]))
print("\nH4 bar-hour drift (UTC+2 server):")
print((stats * [1e4, 1e4, 1, 1]).round(2).rename(
    columns={"mean": "mean_bp", "std": "std_bp"}))

# in-sample split honesty: compute hour ranking on 2015-2019, test 2020+
tr = ret_h4.loc[:"2019"]
te = ret_h4.loc["2020":]
rank = tr.groupby(h4.loc[:"2019"].index.hour).mean().sort_values()
best_hours = list(rank.index[-3:])
print(f"\ntop-3 drift hours fit 2015-19: {best_hours}")
te_hours = te.groupby(h4.loc["2020":].index.hour).mean() * 1e4
print("OOS 2020+ mean bp by hour:", te_hours.round(2).to_dict())
