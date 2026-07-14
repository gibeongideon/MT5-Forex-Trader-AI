"""Campaign 1: individual sleeves + first combined books."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from challenge_lab import *  # noqa

START = "2016-01-01"

# --- sleeves
xau = xau_champ_sleeve().loc[START:]
fx_fast = d1_sleeve(FX, SPEEDS_FAST_D1).loc[START:]
fx_slow = d1_sleeve(FX, SPEEDS_SLOW_D1).loc[START:]
silver_ls = d1_sleeve(["SILVER"], SPEEDS_FAST_D1).loc[START:]
silver_lo = d1_sleeve(["SILVER"], SPEEDS_FAST_D1, long_only=["SILVER"]).loc[START:]
gold_d1 = d1_sleeve(["GOLD"], SPEEDS_FAST_D1, long_only=["GOLD"]).loc[START:]

log("A-xau-champ-h4", stats(xau))
log("B-fx7-fast", stats(fx_fast))
log("B-fx7-slow", stats(fx_slow))
log("C-silver-ls", stats(silver_ls))
log("C-silver-longonly", stats(silver_lo))
log("D-gold-d1-longonly", stats(gold_d1))

# long-window robustness for D1 sleeves (2008+)
log("B-fx7-fast-2008+", stats(d1_sleeve(FX, SPEEDS_FAST_D1)["2008-06-01":]))
log("C-silver-ls-2008+", stats(d1_sleeve(["SILVER"], SPEEDS_FAST_D1)["2008-06-01":]))

# --- correlations
al = pd.DataFrame({"xau": xau, "fx": fx_fast, "ag_ls": silver_ls,
                   "ag_lo": silver_lo}).dropna()
print("\ncorrelations:\n", al.corr().round(2))

# --- combined books (risk weights on ~10%-vol sleeves)
books = {
    "K1-xau60-fx40":          0.60 * al["xau"] + 0.40 * al["fx"],
    "K2-xau50-fx35-agls15":   0.50 * al["xau"] + 0.35 * al["fx"] + 0.15 * al["ag_ls"],
    "K3-xau50-fx30-aglo20":   0.50 * al["xau"] + 0.30 * al["fx"] + 0.20 * al["ag_lo"],
    "K4-xau40-fx40-agls20":   0.40 * al["xau"] + 0.40 * al["fx"] + 0.20 * al["ag_ls"],
    "K5-equal3":              (al["xau"] + al["fx"] + al["ag_ls"]) / 3,
    "K6-xau70-fx30":          0.70 * al["xau"] + 0.30 * al["fx"],
}
for nm, b in books.items():
    log(nm, stats(b))
