"""Campaign 3: final book selection (XAU champion + SILVER champion-recipe)
and FundingPips 2-Step simulation with stress battery."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from challenge_lab import *  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
START = "2016-01-01"

xau = xau_champ_sleeve().loc[START:]
ag = pd.read_csv(os.path.join(HERE, "silver_champ_stream.csv"),
                 parse_dates=["time"], index_col="time").iloc[:, 0].loc[START:]
al = pd.DataFrame({"xau": xau, "ag": ag}).dropna()
print("corr xau/ag:", round(al.corr().iloc[0, 1], 2))

books = {"B-xau100": al["xau"],
         "B-xau85-ag15": 0.85 * al["xau"] + 0.15 * al["ag"],
         "B-xau80-ag20": 0.80 * al["xau"] + 0.20 * al["ag"],
         "B-xau70-ag30": 0.70 * al["xau"] + 0.30 * al["ag"]}
for nm, b in books.items():
    log(nm, stats(b))

# normalize each book to 10% ann vol so k maps to vol target cleanly
def norm10(b):
    return b * (0.10 / (b.std() * np.sqrt(252)))

best = {nm: norm10(b) for nm, b in books.items()}

for nm in ("B-xau100", "B-xau85-ag15", "B-xau80-ag20"):
    print()
    print_sim(nm, best[nm].values, ks=(0.5, 0.7, 0.8, 0.9, 1.0, 1.2))

# --- stress battery on the leading book at the chosen dial
print("\n=== STRESS (book normalized to 10% vol) ===")
lead = best["B-xau85-ag15"]
# 1) weak years only
print_sim("weak-2016-2020", lead.loc[:"2020-12-31"].values, ks=(0.7, 0.8, 0.9))
# 2) intraday-excursion proxy: daily breach at 1.5x the close-to-close move
print_sim("daySafety1.5", lead.values, ks=(0.7, 0.8, 0.9), day_safety=1.5)
# 3) costs x2 (rebuild sleeves at spread_mult=2)
xau2 = xau_champ_sleeve(spread_mult=2.0).loc[START:]
b2 = pd.DataFrame({"x": xau2, "a": ag}).dropna()
lead2 = norm10(0.85 * b2["x"] + 0.15 * b2["a"])
print_sim("costx2(xau leg)", lead2.values, ks=(0.8, 0.9))
# 4) return haircut 25% (edge decay)
hc = lead - lead.mean() * 0.25
print_sim("mean-25pct", hc.values, ks=(0.8, 0.9))

# save the final book stream
lead.to_csv(os.path.join(HERE, "final_book_stream.csv"))
