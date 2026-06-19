"""gold_4h_live.py — forward-test the 2022 XAUUSD 4H turning-point model on a (demo) account.

REGIME-SPECIFIC model (trained 2022+, see train_gold_4h_2022.py / data/GOLD_MTF_4H.md). This is a
DEMO forward-test, not a validated edge. Each new completed 4H bar: rebuild leak-free features,
predict P(up); if flat and P(up)≥thr → BUY, ≤1−thr → SELL, with ATR SL/TP (SL=1×ATR, TP=3×ATR);
force-close after `horizon` 4H bars. One position at a time, tagged by magic.

Modes:
  (default)        connect to the terminal, loop, DRY-RUN (print intended action, NO orders)
  --live           actually place/manage orders on the CONNECTED account (use a DEMO account!)
  --once           single iteration then exit
  --offline        use data/XAUUSD_M15_long.csv instead of the broker (signal check, no connection)

Usage:
  python scripts/gold_4h_live.py --offline --once          # verify the signal path, no broker
  python scripts/gold_4h_live.py --once                    # dry-run vs the live terminal (demo)
  python scripts/gold_4h_live.py --live --lot 0.01         # trade on the connected DEMO account
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.audit_live_champions import add_extra_features
from scripts.backtest_champion_baseline import _load_raw

BUNDLE = ROOT / "data" / "models" / "gold_4h_2022" / "bundle.joblib"
MAGIC = 20220004
ATR_N = 14


def _resample4h(d: pd.DataFrame) -> pd.DataFrame:
    o = d.resample("4h", label="left", closed="left")
    df = pd.DataFrame({
        "open": o["open"].first(), "high": o["high"].max(), "low": o["low"].min(),
        "close": o["close"].last(), "tick_volume": o["tick_volume"].sum(),
        "spread": o["spread"].mean() if "spread" in d else 0.0, "real_volume": 0,
    }).dropna(subset=["open"])
    tr = pd.concat([(df["high"] - df["low"]),
                    (df["high"] - df["close"].shift(1)).abs(),
                    (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_N).mean()
    return df


def _signal(bundle, m15: pd.DataFrame):
    """Return (df4h, P(up) on the last fully-closed 4H bar)."""
    df4 = _resample4h(m15)
    # drop the last bar if its 4H window hasn't closed yet (partial forming bar)
    if len(df4) and (m15.index[-1] < df4.index[-1] + pd.Timedelta("4h") - pd.Timedelta("15min")):
        df4 = df4.iloc[:-1]
    pipe, model, cols = bundle["pipe"], bundle["model"], bundle["cols"]
    X_t, _ = pipe._fp.build(df4, fit=False)
    X_t = add_extra_features(df4, X_t, fix_lookahead=True)
    for c in cols:
        if c not in X_t.columns:
            X_t[c] = 0.0
    X = X_t[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pup = model.predict_proba(X)[:, bundle["up_idx"]]
    return df4, pd.Series(pup, index=X.index)


def _decide(pup_last, thr):
    if pup_last >= thr:
        return "buy", pup_last
    if pup_last <= 1 - thr:
        return "sell", 1 - pup_last
    return None, pup_last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="place/manage orders on the connected account (DEMO!)")
    ap.add_argument("--once", action="store_true", help="single iteration then exit")
    ap.add_argument("--offline", action="store_true", help="use the CSV, no broker connection")
    ap.add_argument("--symbol", default="XAUUSD", help="broker symbol (verify .Z suffix in terminal)")
    ap.add_argument("--lot", type=float, default=0.01)
    ap.add_argument("--poll", type=float, default=300.0, help="seconds between checks")
    args = ap.parse_args()

    b = joblib.load(BUNDLE)
    cfg = b["cfg"]
    thr, k_sl, k_tp, horizon, pip = cfg["thr"], cfg["k_sl"], cfg["k_tp"], cfg["horizon"], cfg["pip"]
    print(f"=== GOLD 4H 2022-model {'LIVE' if args.live else 'DRY-RUN'}{' OFFLINE' if args.offline else ''} ===")
    print(f"  thr={thr} SL={k_sl}×ATR TP={k_tp}×ATR force-close={horizon}×4h  magic={MAGIC}  lot={args.lot}")
    print(f"  ⚠ regime-specific (2022+); intended for DEMO forward-testing.\n")

    conn = None
    if not args.offline:
        from src.core.connector import get_connector
        conn = get_connector("mt5"); conn.connect()
        ai = conn.account_info()
        live_flag = "LIVE-MONEY" if "demo" not in str(getattr(ai, "server", "")).lower() else "demo"
        print(f"  account={getattr(ai,'login','?')} server={getattr(ai,'server','?')} [{live_flag}]")
        if args.live and live_flag == "LIVE-MONEY":
            print("  ⛔ REFUSING --live on a non-demo account. Point the terminal at a DEMO account first.")
            return

    last_bar = None
    while True:
        if args.offline:
            m15 = _load_raw(ROOT / "data" / f"{args.symbol}_M15_long.csv")
        else:
            m15 = conn.get_rates(args.symbol, "M15", 4000)
        df4, pup = _signal(b, m15)
        if pup.empty:
            print("  no signal (insufficient data)");
        else:
            bar_t = pup.index[-1]; p = float(pup.iloc[-1])
            side, conf = _decide(p, thr)
            atr = float(df4["atr"].iloc[-1]); price = float(df4["close"].iloc[-1])
            new_bar = bar_t != last_bar
            print(f"  [{bar_t}] P(up)={p:.3f} → {side or 'flat'} (conf {conf:.2f})  atr={atr:.2f}  px={price:.2f}"
                  f"{'  NEW BAR' if new_bar else ''}")
            if conn is not None:
                pos = conn.get_positions(args.symbol, magic=MAGIC)
                # force-close on horizon timeout
                for pp in pos:
                    age_h = (pd.Timestamp.utcnow().tz_localize(None) - pd.to_datetime(getattr(pp, "time", bar_t), unit="s", errors="coerce")).total_seconds() / 3600 if hasattr(pp, "time") else 0
                    if age_h >= horizon * 4:
                        print(f"    force-close ticket={pp.ticket} (age {age_h:.0f}h)")
                        if args.live: conn.close_position(pp)
                if new_bar and side and not pos:
                    sl = price - k_sl * atr if side == "buy" else price + k_sl * atr
                    tp = price + k_tp * atr if side == "buy" else price - k_tp * atr
                    print(f"    {'PLACING' if args.live else 'WOULD PLACE'} {side.upper()} {args.lot} {args.symbol} "
                          f"SL={sl:.2f} TP={tp:.2f}")
                    if args.live:
                        conn.open_position(args.symbol, side, args.lot, sl=sl, tp=tp,
                                           comment="gold4h2022", magic=MAGIC)
            last_bar = bar_t
        if args.once:
            break
        time.sleep(args.poll)
    if conn is not None:
        conn.disconnect()


if __name__ == "__main__":
    main()
