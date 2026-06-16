"""Download short rates (OECD 3M interbank, monthly) from FRED → data/rates_3m.csv.

Used for the FX carry sleeve: carry(pair) = sign(rate_base - rate_quote). Monthly
series, forward-filled to daily downstream. One series at a time with retries
(network has been flaky; FRED single requests succeed).
"""
from __future__ import annotations
import io, sys, time, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# currency → FRED OECD 3-month interbank rate series
RATE_SERIES = {
    "USD": "IR3TIB01USM156N", "EUR": "IR3TIB01EZM156N", "JPY": "IR3TIB01JPM156N",
    "GBP": "IR3TIB01GBM156N", "AUD": "IR3TIB01AUM156N", "CAD": "IR3TIB01CAM156N",
    "CHF": "IR3TIB01CHM156N", "NZD": "IR3TIB01NZM156N",
}


def _fred(sid: str, retries: int = 4) -> pd.Series | None:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            txt = urllib.request.urlopen(req, timeout=20).read().decode()
            df = pd.read_csv(io.StringIO(txt))
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return df.dropna().set_index("date")["value"]
        except Exception as e:
            print(f"  {sid}: attempt {attempt+1} {str(e)[:50]}", flush=True)
            time.sleep(4)
    return None


def main():
    print(f"=== FRED short rates ({len(RATE_SERIES)} currencies) ===")
    cols = {}
    for ccy, sid in RATE_SERIES.items():
        s = _fred(sid)
        if s is not None and len(s):
            cols[ccy] = s
            print(f"  {ccy} ({sid}): {len(s)} obs  {s.index[0].date()}→{s.index[-1].date()}  last={s.iloc[-1]:.2f}%", flush=True)
        else:
            print(f"  {ccy}: FAILED", flush=True)
    if not cols:
        print("no rates fetched"); sys.exit(1)
    panel = pd.DataFrame(cols).sort_index()
    panel.index.name = "date"
    out = DATA / "rates_3m.csv"
    panel.to_csv(out)
    print(f"\nSaved {panel.shape} → {out}")


if __name__ == "__main__":
    main()
