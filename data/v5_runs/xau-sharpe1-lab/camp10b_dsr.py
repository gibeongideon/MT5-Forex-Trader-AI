"""DSR with the real trial-Sharpe distribution from tonight's log."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from camp10_dsr import daily  # reuses champion daily returns (module runs again)
sys.path.insert(0, "/home/rock/Desktop/2026_Projects/Trader36/MT5")
from src.evaluation.dsr_pbo import deflated_sharpe_ratio

log = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.csv"))
trials = log["sharpe_eval"].astype(float).values / np.sqrt(252)  # per-day units
print(f"n trials tonight: {len(trials)}, mean(ann) {trials.mean()*np.sqrt(252):.2f}, "
      f"max(ann) {trials.max()*np.sqrt(252):.2f}")
d17 = daily.loc["2017":].values
res = deflated_sharpe_ratio(d17, trials)
print("DSR (all tonight's trials):", res)
print(f"benchmark SR annualized: {res['sr_benchmark']*np.sqrt(252):.3f}")
