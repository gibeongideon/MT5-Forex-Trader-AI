"""
download_dukascopy.py — Deep historical M15 from the Dukascopy public datafeed.

Dukascopy serves one LZMA-compressed file per hour of tick data at:
    https://datafeed.dukascopy.com/datafeed/{SYM}/{YYYY}/{MM0}/{DD}/{HH}h_ticks.bi5
where MM0 is the 0-indexed month (January = 00). Each decompressed file is a
sequence of 20-byte big-endian records:
    uint32 ms_offset   (ms since the hour)
    int32  ask         (points)
    int32  bid         (points)
    float32 ask_vol
    float32 bid_vol
Price = int * point_factor (EURUSD 1e-5, USDJPY 1e-3).

This aggregates ticks → M15 OHLCV bars (bid-based OHLC, MT5-style) plus the mean
ask-bid spread in pips, and writes the existing CSV schema:
    time,open,high,low,close,tick_volume,spread,real_volume

Usage:
    python scripts/download_dukascopy.py --symbol EURUSD --from 2015-01-01
    python scripts/download_dukascopy.py --symbol EURUSD --from 2024-01-01 --to 2024-01-08   # smoke test
    python scripts/download_dukascopy.py --symbol USDJPY --from 2015-01-01 --workers 16
"""
from __future__ import annotations

import argparse
import lzma
import struct
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

BASE = "https://datafeed.dukascopy.com/datafeed"
UA = {"User-Agent": "Mozilla/5.0"}

# point factor (price = raw_int * factor) and pip size per symbol
SYMBOLS = {
    "EURUSD": dict(point=1e-5, pip=1e-4),
    "USDJPY": dict(point=1e-3, pip=1e-2),
    "GBPUSD": dict(point=1e-5, pip=1e-4),
    "XAUUSD": dict(point=1e-3, pip=1e-1),
}

_REC = struct.Struct(">Iiiff")   # 20 bytes, big-endian


def _hour_url(sym: str, ts: pd.Timestamp) -> str:
    return f"{BASE}/{sym}/{ts.year:04d}/{ts.month-1:02d}/{ts.day:02d}/{ts.hour:02d}h_ticks.bi5"


def _fetch_hour(sym: str, ts: pd.Timestamp, retries: int = 5) -> bytes | None:
    url = _hour_url(sym, ts)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return b""          # no data this hour (weekend/holiday)
            time.sleep(2 * (attempt + 1))   # backoff on throttle (e.g. 503)
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None                      # persistent failure


def _parse_ticks(raw: bytes, sym: str, ts: pd.Timestamp) -> pd.DataFrame | None:
    if not raw:
        return None
    try:
        data = lzma.decompress(raw)
    except lzma.LZMAError:
        return None
    n = len(data) // 20
    if n == 0:
        return None
    pt = SYMBOLS[sym]["point"]
    ms = np.empty(n, np.int64); bid = np.empty(n); ask = np.empty(n)
    for i in range(n):
        off, a, b, _, _ = _REC.unpack_from(data, i * 20)
        ms[i] = off; ask[i] = a * pt; bid[i] = b * pt
    t0 = ts.value // 10**9
    idx = pd.to_datetime(t0 * 1000 + ms, unit="ms", utc=True).tz_convert(None)
    return pd.DataFrame({"bid": bid, "ask": ask}, index=idx)


def _aggregate_m15(ticks: pd.DataFrame, sym: str) -> pd.DataFrame:
    pip = SYMBOLS[sym]["pip"]
    o = ticks["bid"].resample("15min")
    df = pd.DataFrame({
        "open":  o.first(), "high": o.max(), "low": o.min(), "close": o.last(),
        "tick_volume": ticks["bid"].resample("15min").count(),
        "spread": ((ticks["ask"] - ticks["bid"]).resample("15min").mean() / pip),
    }).dropna(subset=["open"])
    df["real_volume"] = 0
    df["spread"] = df["spread"].round(1)
    return df


def _fetch_many(ex, sym, hrs):
    """Fetch+parse a list of hours; return (tick_frames, failed_hours)."""
    futs = {ex.submit(_fetch_hour, sym, h): h for h in hrs}
    frames, failed = [], []
    for fut in as_completed(futs):
        raw = fut.result()
        if raw is None:
            failed.append(futs[fut]); continue
        tk = _parse_ticks(raw, sym, futs[fut])
        if tk is not None:
            frames.append(tk)
    return frames, failed


def download(sym: str, dt_from: pd.Timestamp, dt_to: pd.Timestamp, workers: int,
             out: Path, chunk_days: int = 7, ckpt_every: int = 25) -> pd.DataFrame:
    """Weekly-batched fetch + aggregate, with periodic checkpointing to `out` and a
    final retry pass over failed hours. Memory bounded (one chunk of ticks held)."""
    all_hours = [h for h in pd.date_range(dt_from, dt_to, freq="h")
                 if h.weekday() != 5]      # skip Saturdays (no Dukascopy data)
    chunk = chunk_days * 24
    n_chunks = (len(all_hours) + chunk - 1) // chunk
    print(f"  {sym}: {len(all_hours):,} hourly files in {n_chunks} chunks  "
          f"{dt_from.date()} → {dt_to.date()}  ({workers} workers)", flush=True)

    bar_frames = []
    failed_all = []
    t0 = time.time()

    def _flush(tag):
        if not bar_frames:
            return
        b = pd.concat(bar_frames).sort_index()
        b = b[~b.index.duplicated(keep="first")]
        b.index.name = "time"
        b[["open","high","low","close","tick_volume","spread","real_volume"]].to_csv(out)
        print(f"    [checkpoint {tag}] wrote {len(b):,} bars → {out}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ci in range(n_chunks):
            hrs = all_hours[ci * chunk:(ci + 1) * chunk]
            frames, failed = _fetch_many(ex, sym, hrs)
            failed_all += failed
            if frames:
                ticks = pd.concat(frames).sort_index()
                bar_frames.append(_aggregate_m15(ticks, sym))
            if (ci + 1) % 10 == 0 or ci == n_chunks - 1:
                nb = sum(len(b) for b in bar_frames)
                print(f"    chunk {ci+1}/{n_chunks}  bars={nb:,}  "
                      f"failed_hrs={len(failed_all):,}  {(time.time()-t0)/60:.1f}min", flush=True)
            if (ci + 1) % ckpt_every == 0:
                _flush(f"chunk {ci+1}")

        # retry pass over failed hours (network throttle recovery)
        if failed_all:
            print(f"    retry pass: {len(failed_all):,} failed hours...", flush=True)
            still = failed_all
            for rnd in range(2):
                if not still:
                    break
                frames, still = _fetch_many(ex, sym, still)
                if frames:
                    ticks = pd.concat(frames).sort_index()
                    bar_frames.append(_aggregate_m15(ticks, sym))
                print(f"      round {rnd+1}: recovered {len(frames)}, still failed {len(still)}",
                      flush=True)
            failed_all = still

    if not bar_frames:
        print("    no tick data fetched"); return pd.DataFrame()
    if failed_all:
        print(f"    WARNING: {len(failed_all):,} hours permanently failed — minor gaps")
    bars = pd.concat(bar_frames).sort_index()
    bars = bars[~bars.index.duplicated(keep="first")]
    print(f"    → {len(bars):,} M15 bars  spread(pips) mean={bars['spread'].mean():.2f} "
          f"median={bars['spread'].median():.2f}")
    return bars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD", choices=list(SYMBOLS))
    ap.add_argument("--from", dest="dfrom", default="2015-01-01")
    ap.add_argument("--to", dest="dto", default=None, help="default: today")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=None, help="default data/{SYM}_M15_long.csv")
    args = ap.parse_args()

    dt_from = pd.Timestamp(args.dfrom)
    dt_to   = pd.Timestamp(args.dto) if args.dto else pd.Timestamp.utcnow().tz_localize(None).floor("h")
    out = Path(args.out) if args.out else DATA_DIR / f"{args.symbol}_M15_long.csv"

    out.parent.mkdir(exist_ok=True)
    print(f"\n{'='*68}\n  DUKASCOPY DEEP HISTORY — {args.symbol}\n{'='*68}")
    bars = download(args.symbol, dt_from, dt_to, args.workers, out)
    if bars.empty:
        print("No data. Aborting."); sys.exit(1)

    bars.index.name = "time"
    bars = bars[["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]
    out.parent.mkdir(exist_ok=True)
    bars.to_csv(out)
    print(f"\n  Saved {len(bars):,} bars → {out}")
    print(f"  Range: {bars.index[0]} → {bars.index[-1]}")
    print("Done.")


if __name__ == "__main__":
    main()
