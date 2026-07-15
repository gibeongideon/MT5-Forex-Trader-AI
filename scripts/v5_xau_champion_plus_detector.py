"""Does the ML turning-point detector ADD value on top of the long-only champion?

Both on the SAME H4 timeline. Champion = camp8 finalist (conc1.5, buffer 0.2),
long-only vol-targeted EWMAC+breakout (eval Sharpe ~0.97). Detector = HistGBoost
bottom/top model, TRAINED on data < 2017 and applied 2017+ (so the overlay is
out-of-sample over the exact window the champion is judged on).

Overlays modify the champion position:
    boost     pos*(1 + a*b_bot)            add exposure near detected bottoms
    trim      pos*(1 - c*b_top)            cut exposure near detected tops
    both      pos*(1 + a*b_bot)*(1 - c*b_top)
Net Sharpe is INVARIANT to a constant leverage scale (cost is linear in position),
so comparing champion vs overlay Sharpe is a pure TIMING test — no leverage cheat.
CAGR is also shown after rescaling each overlay to the champion's avg exposure.

    python scripts/v5_xau_champion_plus_detector.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = "/home/rock/Desktop/2026_Projects/Trader36/MT5"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data/v5_runs/xau-longonly-champion"))

from xau_lab import load_h4, ewmac_fc, breakout_fc, ANN_H4, SLIP_USD, EVAL_START  # noqa
from scripts.v5_xau_turning_ml import zigzag_swings, features, label_near, atr  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

TVOL, VOL_HL, MAXLEV, BUF = 0.10, 42, 8.0, 0.2


def champion_forecast(h4):
    D = 6
    MID = tuple((f * D, s * D) for f, s in ((16, 64), (32, 128), (64, 256)))
    base = ewmac_fc(h4["close"], MID)
    bko_f = breakout_fc(h4["close"], [d * D for d in (10, 20, 40)])
    L = lambda s: s.clip(lower=0.0)
    maxewbko = np.maximum(L(base), L(bko_f))
    norm = lambda s: s * (1.0 / s.abs().expanding(min_periods=120).mean().shift(1))
    conc = lambda s, p: norm(s.clip(lower=0.0) ** p)
    return (conc(maxewbko, 1.5) * 0.8 + 0.15).clip(0, 2)


def raw_position(h4, fc):
    """Vol-targeted position with causal buffer band (camp8 exact engine, pre-delay)."""
    close = h4["close"]; ret = close.pct_change()
    vol = ret.ewm(halflife=VOL_HL, min_periods=20).std() * np.sqrt(ANN_H4)
    pos = (fc * (TVOL / vol)).clip(-MAXLEV, MAXLEV)
    avg = (TVOL / vol).clip(0, MAXLEV)
    band = BUF * avg
    p, out, held = pos.values, np.zeros(len(pos)), 0.0
    for i in range(len(p)):
        if np.isfinite(p[i]):
            b = band.iloc[i] if np.isfinite(band.iloc[i]) else 0.0
            if abs(p[i] - held) > b:
                held = p[i] - np.sign(p[i] - held) * b
        out[i] = held
    return pd.Series(out, index=pos.index)


def metrics(pos_pre, h4):
    """pos_pre = position decided at bar close; apply next bar (delay 1). Net of cost."""
    close = h4["close"]; ret = close.pct_change()
    pos = pos_pre.shift(1).fillna(0.0)
    cost_frac = (h4["spread_px"] / 2.0 + SLIP_USD) / close
    net = (pos * ret - pos.diff().abs().fillna(0.0) * cost_frac).fillna(0.0)
    eq = (1 + net).cumprod()
    out = {}
    for tag, sl in (("full", slice(None)), ("eval", slice(EVAL_START, None))):
        e = eq.loc[sl]; e = e / e.iloc[0]
        dd = float((e / e.cummax() - 1).min() * 100)
        dl = e.resample("D").last().pct_change(fill_method=None).dropna()
        sh = float(dl.mean() / dl.std() * np.sqrt(252)) if dl.std() > 0 else 0.0
        yrs = (e.index[-1] - e.index[0]).days / 365.25
        cagr = float(e.iloc[-1] ** (1 / yrs) - 1) * 100 if yrs > 0 else 0.0
        out[tag] = (sh, dd, cagr)
    out["turn"] = float(pos.diff().abs().sum() / ((pos.index[-1] - pos.index[0]).days / 365.25))
    out["expo"] = float(pos.loc[EVAL_START:].abs().mean())
    return out


def train_detector(h4, order, theta_mult, tol):
    theta = theta_mult * atr(h4)
    sells, buys = zigzag_swings(h4, order, theta)
    f = features(h4)
    X = f.values
    ok = ~np.isnan(X).any(axis=1)
    is_train = np.asarray(h4.index < pd.Timestamp(EVAL_START)) & ok
    probs = {}
    for name, truth in (("bot", buys), ("top", sells)):
        y = label_near(truth, len(h4), tol)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                             max_depth=4, l2_regularization=1.0, random_state=7)
        clf.fit(X[is_train], y[is_train])
        p = np.full(len(h4), np.nan)
        p[ok] = clf.predict_proba(X[ok])[:, 1]
        probs[name] = pd.Series(p, index=h4.index).fillna(0.0)
    return probs, len(buys), len(sells)


def show(name, m):
    se, dde, ce = m["eval"]; sf = m["full"][0]
    print(f"  {name:22} eval SR {se:+.3f}  full SR {sf:+.3f}  DD {dde:6.1f}%  "
          f"CAGR {ce:+6.1f}%  turn {m['turn']:6.1f}  expo {m['expo']:.2f}")


def main():
    h4 = load_h4()
    fc = champion_forecast(h4)
    base_pos = raw_position(h4, fc)
    probs, nb, ns = train_detector(h4, order=5, theta_mult=1.5, tol=3)
    b_bot, b_top = probs["bot"], probs["top"]

    print(f"[champion + detector] H4 bars={len(h4)}  swings {nb} buys / {ns} sells  "
          f"detector trained <{EVAL_START}, applied 2017+ (OOS)")
    m_base = metrics(base_pos, h4)
    print("\n--- baseline ---")
    show("champion (conc1.5)", m_base)
    tgt_expo = m_base["expo"]

    def rescaled(pos):
        e = pos.loc[EVAL_START:].abs().mean()
        return pos * (tgt_expo / e) if e > 0 else pos

    print("\n--- overlays (rescaled to champion avg exposure) ---")
    for a in (0.5, 1.0, 2.0):
        show(f"boost bot a={a}", metrics(rescaled(base_pos * (1 + a * b_bot)), h4))
    for c in (0.5, 1.0):
        show(f"trim top c={c}", metrics(rescaled(base_pos * (1 - c * b_top)), h4))
    for a, c in ((1.0, 0.5), (2.0, 1.0)):
        show(f"both a={a} c={c}", metrics(rescaled(base_pos * (1 + a * b_bot) * (1 - c * b_top).clip(0)), h4))

    # reference: detector-only long (bottom prob as the whole signal)
    print("\n--- reference ---")
    det_only = (b_bot * (TVOL / (h4['close'].pct_change().ewm(halflife=VOL_HL, min_periods=20).std()
                                 * np.sqrt(ANN_H4)))).clip(0, MAXLEV)
    show("detector-only long", metrics(rescaled(det_only), h4))


if __name__ == "__main__":
    main()
