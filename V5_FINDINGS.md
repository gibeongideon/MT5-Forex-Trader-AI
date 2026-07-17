# V5 FINDINGS — settled experiments (do not repeat)

Running ledger of XAUUSD research that has been **run, measured, and closed**. Each
entry: what was tried, the honest result, and the verdict. If an idea here is
marked DISPROVEN / DEAD, do not re-run it without a materially new angle.

Conventions: net Sharpe from daily-resampled equity × √252; eval window = 2017+;
the live HFM cent account's true gold cost is **$0.34 spread** (the `XAUUSD_*_long.csv`
spread column understates by 10× — use `--fixed-spread-usd 0.34`, not the raw column).

---

## Champions currently trusted (positive net edge)

| Strategy | Net edge | Status | Notes |
|---|---|---|---|
| Sharpe-1.6 drift portfolio | eval SR **1.59**, DSR 0.9998 | backtested, **not deployed** | champion recipe across BTC/indices/XAU/silver + LS diversifiers; the real upside |
| Long-only XAU champion | eval **0.99–1.04**, live ~0.97 | **LIVE** (acct 360542) | H4, vol-targeted EWMAC+breakout, conc^1.5, long-only. `data/v5_runs/xau-longonly-champion/` |
| LS ensemble | SR **0.81** | **LIVE** (acct 360541) | long-short trend diversifier |

Single-XAU ceiling ≈ 1.06; the jump to 1.6 is multi-asset diversification only.

### FundingPips challenge — DIVERSIFY the book (2026-07-15)
Single-XAU passes the FP 2-Step only ~61% once the daily-loss rule is measured on *floating* P&L (day_safety=1.5 proxy) — all risk in one asset breaches the 5% intraday line ~32% of the time. Diversifying the same champion recipe across the tradeable drift classes {eq_us SPX/NDX/DJI, eq_eu DAX/FTSE/STOXX, eq_ap NIKKEI/ASX, crypto BTC/ETH, xau H4-champ, metal SILVER}, equal-class-risk at 9% vol, lifts realistic pass to **~80% (daily-loss fails 32%→7%), median ~9.4mo**. (Full basket incl. rates/energy = 93% but those aren't offered.) Engine `scripts/v5_basket_challenge.py` (`--backtest` validates eval SR 1.27 / 80% pass; `--targets` emits live per-symbol leverage). Live MT5 order-wiring deferred until the FP account is purchased (needs real symbol names + terminal). Single-XAU `v5_xau_challenge.py` remains the fallback. Detail in CHALLENGEBOT.MD / memory.

### Basket improvement experiments (2026-07-16, `data/v5_runs/basket-ls-experiment/`)
- **Long/short per sleeve (like the cent `ls` bot) — DISPROVEN, badly:** eval SR 1.26→0.23, pass 92%→48%. Drift assets trend UP; shorting them bleeds. Crash-hedge (shorts only in deep downtrends) also worse (0.97). Do NOT add shorts.
- **Sharpe-weighting sleeves = lookahead illusion:** 1.46/96% in-sample, but walk-forward (past-data weights) = 1.25/88.7% ≈ equal-class. Equal-class weighting is already near-optimal; keep it.
- **GENUINE robust win: portfolio-level VOLATILITY TARGETING** (scale book to constant trailing vol, causal). Eval SR **1.26→1.39**, +dd-scaler **→1.43**; pass 91.9%→94.3% @7% (96.8% @6%), and FASTER. Recommended upgrade — scale per-symbol target leverages by a trailing-vol scalar on the book's own returns. Report: `basket-ls-experiment/REPORT.md`.

---

## DISPROVEN / DEAD (do not re-run)

### 1. M15 next-bar fade (mean-reversion) — DEAD net of spread
- **Signal:** fade extreme closes (close near bar LOW → long next bar; near HIGH → short). `scripts/v5_xau_fade_backtest.py`, live paper `scripts/v5_xau_fade_paper.py`.
- **Gross edge is real:** zero-cost M15 hours{8,20,22} FLAT ≈ Sharpe +1.9–2.3, win 54.8%, consistent every year.
- **Dies net of the true $0.34 spread:** win 36%, Sharpe **−6.8**, total −90% over 11.4 yrs. Deployed hours-restricted config included. The per-trade edge is a few cents; the spread is ~10× it.
- **Break-even spread ≈ $0.04–0.07** (need institutional/raw). HFM Zero/Raw ~$0.10–0.15 all-in is still above it.
- **Live paper +$10 over 2 weeks was a lucky window** — 11-yr net is −18%/yr.
- **Verdict:** not deployable as a spread-crossing taker at any timeframe (M15/M30/H1/H4 all tested). Only theoretical path = maker/limit fills that *earn* the spread (untested, likely infeasible retail).

### 2. Martingale / anti-martingale overlays on the fade — DEAD (ruin machine)
- `scripts/v5_xau_fade_martingale.py` — engines flat / double4 (classic capped-4 doubling) / recover4 (deficit-targeted, ladder capped at 4 trades, reset on recovery).
- **On the NET signal (36% win): recover4 BUSTS 100% of 500 block-bootstrap paths, ruin by ~trade 1,266.** double4 −61%, flat −23%.
- **On a GROSS positive-edge signal (54.8% win): recover4 = +152% vs flat +6% / double4 +13%, 0% bust, DD 11%** — proves the martingale is a *lever*, not an edge-creator.
- **Verdict:** decision variable is solely sign(net edge). Base fade is net-negative → no martingale rescues it. Do not deploy any progressive sizing on a negative-edge base.

### 3. Turning-point DETECTION (buy/sell reversal accuracy) — real skill, NOT tradeable
- `scripts/v5_xau_turning_points.py` (accuracy), `scripts/v5_xau_turning_ml.py` (precision push), `scripts/v5_xau_turning_trade.py` (trade sim), `scripts/v5_xau_champion_plus_detector.py` (combine). Ground truth = ZigZag swings; all detectors causal; scored precision/recall/F1 vs a random-coverage baseline.
- **Detection works above chance, robustly (no decay to 2024+):** best rule = Bollinger z-score, buy precision ~50–58% vs 33.5% random (lift 1.5–1.9×). **Bottoms detect materially better than tops** (V-panics vs rounded tops) — every config. Fade candle-shape = **zero** turning skill (lift 1.00×).
- **70% precision IS reachable OOS** with a HistGradientBoosting model — **BUY/bottoms tol=±3 → 71% precision @ 22% recall (2.1× random), 75%@10%, 80%@5%, PR-AUC 0.60.** SELL/tops cannot (70% only @ ~9% recall). Rules alone cap ~62%. Top features: 5-bar ATR-momentum, rank-in-window, z-score, RSI7. CAVEAT: judge by LIFT — widening tol inflates absolute precision by raising the random base.
- **But precision ≠ P&L.** Standalone bottom-long (~27 flags/mo, ~6–16 trades/mo): only profitable exit was hold-48-bars (+35%, SR 0.81) which merely harvests up-drift and **LOST to buy&hold gold (+113%)** in the same 2024–26 bull window. Short holds flat-to-negative.
- **Combining with the champion FAILS.** Overlaying the OOS detector on the H4 champion at equal exposure (pure timing test): champion alone eval SR **0.99** → every overlay worse (mildest bottom-boost 0.88, strong 0.66; trim-tops 0.88/0.65; both 0.40; detector-only-long −0.02). Turnover explodes 9.8→64→427 (prob wobble churns spread). Reason: champion is TREND-following (more exposure after breakouts), detector is MEAN-REVERSION (buy dips) — antagonistic; reversion-timing dilutes the trend edge and pays cost.
- **Anti-martingale ('untimartingale') sizing on the detector trades DOESN'T help** (`scripts/v5_xau_detector_antimartingale.py`). Best base (hold-48) flat Sharpe 0.84 → anti 0.30–0.53 at 2–3× the drawdown; hold-12 anti clearly negative (SR −0.5 to −0.8). Root cause: **win/loss lag-1 autocorrelation ≈ 0 (+0.05 to −0.03) — wins do not cluster.** Progressive sizing (anti *or* martingale) only adds value when outcomes are serially correlated; here trades are ~independent, so pressing winners just adds variance with no expected-return gain → Sharpe falls. Sizing cannot create edge on a streak-less, buy&hold-losing base.
- **Verdict:** detector is real science but not a tradeable edge and does not improve the trend bot. Leave the champion alone (it handles entry via vol-targeting + buffer). Turning-point timing and trend-following are orthogonal-to-antagonistic.

### 3b. Fast (non-trend) strategy exploration (2026-07-16, `data/v5_runs/fast-strategies/`)
Wide sweep for faster, consistent, non-trend edges, all net of spread, OOS=2021+.
- **Cross-sectional reversal on GLOBAL indices = FAKE** (Sharpe 3.5): non-synchronous close times (NIKKEI/ASX close hours before US) leak future info. US-only synchronous = −0.37. Classic artifact — reject any cross-sectional signal on assets with different close times.
- Standalone fast edges are WEAK net of cost: 1-day reversal best NDX 0.65; turn-of-month = rest-of-month (no distinct edge); day-of-week tiny.
- **THE FINDING — diversified FAST ENSEMBLE (overnight + short-term reversal, ~26 daily signals, avg corr 0.02):** select on 2017-2020, deploy 2021+ OOS → **OOS Sharpe 1.30** (IS 1.32, held), consistent (2022 only −0.4). Decompose OOS: **overnight-only 1.19** (the driver, but needs LIVE fill check — close→open gap may not be tradeable on 24h CFDs), **reversal-only 0.57** (cleanly tradeable close→close, ~0 corr to trend). **TREND+FAST 50/50 = 1.44** vs trend alone 1.17 → real diversifying lift. No get-rich-quick fast scheme; the ensemble is a genuine COMPLEMENT to trend, not a replacement.
- **VERIFIED DEAD on HFM's real tradeable instruments (2026-07-16, `scripts/v5_overnight_verify.py` + `v5_fast_verify_all.py` vs demo 57482374):** overnight ensemble −4.65 (backtest used CASH-index close 16:00→open 09:30 window; HFM's futures-CFDs break at a different time → no premium, slightly negative); intraday +1.22 but 0.97 corr to buy-hold = JUST DRIFT; reversal R1/R2/R5 = +0.13/+0.02/−0.35 = no edge (backtest 0.57 doesn't hold on real closes/spreads). **NOTHING in the fast family is deployable.** The backtest edges died on: bar-boundary shifts, wider real spreads, and drift-in-disguise. **RULE: always verify a fast edge on the BROKER's own symbols before building — cash/downloaded data lies about overnight windows and spreads.** Durable edge stays the TREND/DRIFT book. Report: `data/v5_runs/fast-strategies/REPORT.md`.

### 3c. Fast / intraday TREND runner (2026-07-17, `data/v5_runs/fast-trend/`)
Hunt for a *short-term trend* bot to complement the slow H4 champion (more
trades/day). Distinct from the dead fade/reversal work — this is trend, not
mean-reversion. Engines: `scripts/v5_xau_fast_trend_lab.py` (vectorized
vol-target sweep, buffered, combined-book vs champion) + `v5_xau_fast_trend_discrete.py`
(real lot/stop engine, monkeypatched fast champion signal).
- **Long-only beats LS at every speed** (kill-the-shorts again). Trend edge is
  gross-positive at ALL speeds (breakout gross SR ≤1.3) but **net Sharpe falls
  with turnover — spread is the tax.** No intraday-specific alpha: **session-ORB
  and intraday-momentum are gross-positive, net-DEAD** (same failure as fade).
- **Carver no-trade buffer cuts turnover 3–4× at ~no net-Sharpe loss** — key
  lever for spread-bound fast books; but slowing it down raises corr-to-champion
  to 0.73–0.80.
- **A fast sleeve does NOT improve the combined book:** champ-alone 1.28 →
  best 50/50 combo 1.23. On one asset a faster book is a correlated weaker clone.
- **Discrete reality kills the vectorized illusion:** net SR ~1.0 (continuous)
  → **0.4–0.5** (real 3×ATR stops whipsaw at 38% win, $0.34 spread, quantized
  lots); **negative 2017/2018/2021, edge only in the 2024–26 bull** = leveraged
  bull-beta, not robust.
- **The killer is the cent spread — account-type-fixable.** M30 fast champion,
  ~19 trades/mo: **$0.34 → 0.50 (dead); $0.12 raw/ECN → 0.85 (deployable);
  $0.02 → 1.07.** Halving spread ~doubles net Sharpe. M30 fast = best config.
- **LIVE VPS-VERIFIED (2026-07-17, read-only bridge probe):** live cent XAUUSDc
  spread measured **$0.36** (not $0.34 — slightly worse) → discrete net SR 0.49,
  DEAD confirmed. Raw-tier gold IS real at HFM (**XAUUSDb $0.10 FULL-tradable on
  demo**, discrete 0.89; XAUUSDr $0.16 disabled) but **NOT visible on the live
  cent account group** → no tight-spread path on the current live account. Gold
  swap −$0.72/oz/night long (unmodeled). Probes: `scripts/vps_spread_probe.py`,
  `vps_symbol_tradability.py`.
- **Verdict:** NOT deployable on the cent account (verified $0.36). Viable as an
  *activity* play only on a raw-tier HFM gold account (~$0.10–0.16, e.g. XAUUSDb),
  which the live cent group can't reach; and even there it doesn't lift a
  champion+fast portfolio (correlated). Real lever for more trades = cross-ASSET
  trend (BTC/NDX), not cross-speed on XAU. Report: `data/v5_runs/fast-trend/REPORT.md`.

### 4. Earlier disproven overlays (see memory for detail)
- **Per-trade probability sizing / meta-labeling** — fails twice; vol-targeting only cuts drawdown, adds no return.
- **Gold-silver spread** — corr 0.79 but z-spread edge is pre-2015-only, dead OOS 2017+.
- **Session / regime / carry / RL single-XAU overlays** — none beat the plain trend champion.

---

## Operational notes
- Live dual bots reconcile hourly via `xau-dual` user timer; the systemd service intermittently marks `failed` on a hung `winedevice.exe` at teardown (trading completes & exits 0 first) — cosmetic but leaves orphan wine procs; teardown fix pending.

_Last updated 2026-07-15._
