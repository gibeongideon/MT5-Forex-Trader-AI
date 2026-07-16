# Basket improvement experiments — 2026-07-16

Explored whether the FundingPips basket (equal-class LONG-ONLY champion, eval SR 1.26,
92% pass @7% vol) can be improved. Scripts: `ls_experiment.py`, `improve2/3/4.py`.

## User hypothesis: long/short (LS) per sleeve like the cent `ls` bot — DISPROVEN
| Variant | eval SR | maxDD | FP pass |
|---|---|---|---|
| Long-only (current) | **1.26** | −15.9% | **92%** |
| Long/short | 0.23 | −26.7% | 48% (failDD 52%) |
| 50/50 blend | 0.90 | −21.0% | 86% |

These are DRIFT assets (indices/gold/crypto/silver) that trend UP long-term. Letting
each sleeve short = betting against the drift → the short side bleeds systematically.
Confirms "kill-the-shorts is the lever" at portfolio level. The cent `ls` bot works on
XAU only as a *diversifier* (SR 0.81 < champ 0.97), not because shorting is better.

## Signal tweaks — none robustly help
- **crash-hedge** (shorts only in deep downtrends): 0.97 SR — shorts hurt even as tail hedge.
- **faster speeds**: 1.26 SR, deeper DD — no gain.
- **Sharpe-weight sleeves**: 1.46 SR / 96% pass IN-SAMPLE, but **walk-forward (past-data
  weights) = 1.25 / 88.7%** — pure lookahead illusion. Equal-class is already near-optimal.
- **drop weak eq_eu/eq_ap**: 1.39 SR but concentration lowers pass (89%). Not clearly better.

## GENUINE robust win: portfolio-level VOLATILITY TARGETING
Scale the whole book to constant trailing vol (EWMA halflife 20, shifted 1d — no lookahead).
Optionally add a drawdown-scaler (de-risk as running DD deepens).

| Variant | eval SR | 7% vol pass | 6% vol pass | median |
|---|---|---|---|---|
| BASE equal-class (current) | 1.26 | 91.9% | 94.7% | 13.7mo |
| **+ vol-target** | **1.39** | 93.1% | 95.9% | 12.4mo (faster) |
| **+ vol-target + dd-scaler** | **1.43** | 94.3% | **96.8%** | 12.5mo |

Robust (causal), well-founded (standard CTA technique), improves Sharpe **+10–13%**,
raises pass rate, AND passes faster. **This is the recommended upgrade.**

## Vol-dial frontier (return vs pass-safety), BASE
5%→97% pass/21mo · 6%→94.7%/17mo · **7%→91.9%/13.7mo (current)** · 8%→81.7%/11mo.

## RECOMMENDATION
1. **Add portfolio vol-targeting (+dd-scaler)** to the executor → SR 1.26→1.43, pass
   ~94% @7% or ~97% @6%. Only robust improvement found. Modest code change (scale the
   per-symbol target leverages by a trailing-vol scalar computed on the book's own returns).
2. **Do NOT add shorts** — disproven, badly.
3. **Keep equal-class weighting** — Sharpe-weighting is overfitting.
4. Vol dial: 7% (balanced) or 6% (safer/slower) — both fine.
