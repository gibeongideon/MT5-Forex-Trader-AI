import shutil, os, glob
SRC = os.path.dirname(os.path.abspath(__file__))
DST = "/home/rock/Desktop/2026_Projects/Trader36/MT5/data/v5_runs/xau-sharpe1-lab"
os.makedirs(DST, exist_ok=True)
for f in glob.glob(f"{SRC}/*.py") + [f"{SRC}/results.csv"]:
    shutil.copy2(f, DST)
print("copied:", sorted(os.listdir(DST)))
