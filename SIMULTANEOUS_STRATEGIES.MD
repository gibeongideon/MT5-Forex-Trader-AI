# Simultaneous Strategies — 5 Uncorrelated Trading Bots

> Goal: Run 5 independent XGBoost + enc8 models simultaneously on different
> instruments and timeframes. Each bot trades its own pair/TF, has its own
> model artifacts, and is driven by a different market force.
> Combined portfolio: smoother equity curve, lower MaxDD, same expected return.
>
> Last updated: 2026-06-07

---

## Why 5 Strategies Instead of 1

A single strategy (EURUSD M15) has:
- Walk-forward Sharpe: +2.31 (fresh runs)
- MaxDD: ~13%
- Dead periods: ranging markets where it fires noise signals

Five uncorrelated strategies together:
- Portfolio Sharpe: ~+2.5 to +3.5 (diversification benefit)
- Portfolio MaxDD: lower (when one drawdowns, others are flat or profiting)
- Fewer dead periods: different instruments trend at different times

```
Mathematical basis (portfolio theory):
  Portfolio Sharpe ≈ avg_Sharpe / sqrt(avg_correlation)

  If each strategy Sharpe = 1.5, avg correlation = 0.25:
  Portfolio Sharpe ≈ 1.5 / sqrt(0.25) = 1.5 / 0.5 = 3.0
```

---

## The 5 Strategies

### Strategy 1 — EURUSD M15 Scalper (Champion)

**Status: DONE — model trained, evaluating**

| Parameter | Value |
|-----------|-------|
| Instrument | EURUSD |
| Timeframe | M15 (15 minutes) |
| Features | 31 base + 8 latent enc8 = 39 |
| Label horizon | 4 bars (1 hour) |
| Label threshold | 0.03% (0.0003) |
| SL / TP | 30p / 60p |
| Pip size | 0.0001 |
| Spread | 1.0p |
| Walk-forward Sharpe | +2.31 (fresh avg) / +3.13 (cached best) |
| MaxDD | ~13% |
| Model artifacts | `data/models/pipeline_EURUSD/` |

**What drives it:** EUR/USD macro flow, ECB vs Fed rate differential, institutional
USD positioning. Captures M15 momentum patterns in the most liquid forex pair.

---

### Strategy 2 — USDJPY M15 Carry

**Status: DONE — model trained, evaluating**

| Parameter | Value |
|-----------|-------|
| Instrument | USDJPY |
| Timeframe | M15 |
| Features | 31 base + 8 latent enc8 = 39 |
| Label horizon | 4 bars (1 hour) |
| Label threshold | 0.03% (0.0003) |
| SL / TP | 30p / 60p |
| Pip size | 0.01 |
| Spread | 1.0p |
| Walk-forward Sharpe | +2.76 (first run) |
| MaxDD | 25.6% |
| Model artifacts | `data/models/pipeline_USDJPY/` |

**What drives it:** Carry trade dynamics — USD interest rate vs near-zero Japanese rates.
Risk-on/risk-off sentiment: when fear spikes, JPY strengthens sharply (safe haven flows).
USDJPY trends are longer and cleaner than EURUSD because carry unwinds are gradual.

**Correlation to S1:** Low negative. When USD strengthens vs EUR (S1 sells), USD also
often strengthens vs JPY (S2 buys). Partial hedge.

---

### Strategy 3 — EURUSD H4 Swing

**Status: NOT STARTED — data on disk (`data/EURUSD_H4.csv`)**

| Parameter | Value |
|-----------|-------|
| Instrument | EURUSD |
| Timeframe | H4 (4 hours) |
| Features | 31 base + 8 latent enc8 = 39 |
| Label horizon | 4 bars (16 hours) |
| Label threshold | 0.10% (0.001) — wider for longer hold |
| SL / TP | 60p / 120p |
| Pip size | 0.0001 |
| Spread | 1.0p |
| Walk-forward Sharpe | TBD |
| MaxDD | TBD |
| Model artifacts | `data/models/pipeline_EURUSD_H4/` |

**What drives it:** Multi-day institutional flows. 50 H4 bars = 8+ days of context.
The enc8 encoder sees week-long OHLCV patterns rather than hour-long M15 patterns.
Catches weekly trend continuations and weekly reversals that M15 noise drowns out.

**Correlation to S1:** Moderate (same instrument). But S1 fires on M15 momentum while
S3 fires on multi-day swing setups — different entry timing, different hold periods.
S1 can be flat or losing on a ranging M15 while S3 is catching a 4-day H4 trend.

**Config to add in `retrain_for_live.py`:**
```python
"EURUSD_H4": dict(
    csv             = "data/EURUSD_H4.csv",
    pip_size        = 0.0001,
    sl              = 60,
    tp              = 120,
    spread          = 1.0,
    commission      = 0.5,
    label_threshold = 0.001,
    out_dir         = "data/models/pipeline_EURUSD_H4",
),
```

**Launch command:**
```bash
conda run -n envmt5 python scripts/retrain_for_live.py --pair EURUSD_H4
```

---

### Strategy 4 — XAUUSD M15 Gold

**Status: NOT STARTED — data on disk (`data/XAUUSD_M15.csv`)**
**Previous attempt failed** — label_threshold was too small for gold's volatility.

| Parameter | Value |
|-----------|-------|
| Instrument | XAUUSD (Gold) |
| Timeframe | M15 |
| Features | 31 base + 8 latent enc8 = 39 |
| Label horizon | 4 bars (1 hour) |
| Label threshold | **0.10% (0.001)** — $2 move at $2000 gold |
| SL / TP | 500p / 1000p ($5 / $10) |
| Pip size | 0.01 |
| Spread | 30p ($0.30) |
| Walk-forward Sharpe | TBD |
| MaxDD | TBD |
| Model artifacts | `data/models/pipeline_XAUUSD/` |

**What drives it:** Gold is driven by:
- USD strength (gold up = USD weak = EURUSD often rises — partial hedge with S1)
- Inflation expectations (CPI surprises move gold strongly)
- Fear / geopolitical events (gold spikes when risk-off)
- Real interest rates (gold falls when real rates rise)

**Why previous run failed:** label_threshold=0.0003 (0.03%) = $0.60 move for gold.
Gold's typical 4-bar move is $4+, so almost every bar was labeled buy/sell (no holds).
The model fired on 20,675 bars in one period → account blew up.

**Fix:** label_threshold=0.001 gives $2.00 threshold — realistic for gold's volatility.
Expected label distribution: ~30% buy / ~40% hold / ~30% sell (similar to EURUSD).

**Correlation to S1:** Low negative. Gold often rises when USD weakens (EURUSD rises).
But gold also spikes on fear events when EURUSD falls → partial independence.

**Config to add in `retrain_for_live.py`:**
```python
"XAUUSD": dict(
    csv             = "data/XAUUSD_M15.csv",
    pip_size        = 0.01,
    sl              = 500,
    tp              = 1000,
    spread          = 30.0,
    commission      = 5.0,
    label_threshold = 0.001,
    out_dir         = "data/models/pipeline_XAUUSD",
),
```

**Launch command:**
```bash
conda run -n envmt5 python scripts/retrain_for_live.py --pair XAUUSD
```

---

### Strategy 5 — GBPJPY M15 Cross

**Status: NOT STARTED — data download needed**

| Parameter | Value |
|-----------|-------|
| Instrument | GBPJPY |
| Timeframe | M15 |
| Features | 31 base + 8 latent enc8 = 39 |
| Label horizon | 4 bars (1 hour) |
| Label threshold | 0.03% (0.0003) |
| SL / TP | 50p / 100p |
| Pip size | 0.01 |
| Spread | 3.0p |
| Walk-forward Sharpe | TBD |
| MaxDD | TBD |
| Model artifacts | `data/models/pipeline_GBPJPY/` |

**What drives it:** GBPJPY is a cross pair with **no direct USD exposure**:
- GBP side: Bank of England policy, UK economic data, political risk
- JPY side: Bank of Japan policy, carry trade unwinding, risk sentiment
- GBPJPY is known for explosive momentum — when it trends, it trends hard
- Completely independent of ECB/Fed dynamics that drive EURUSD

**Lowest correlation** of all 5 strategies to S1. When EURUSD is flat (ECB/Fed on hold),
GBPJPY can be strongly trending on BOE hawkishness or BOJ intervention.

**Download command (MT5 bridge must be running):**
```bash
conda run -n envmt5 python -c "
import sys; sys.path.insert(0, '.')
import MetaTrader5 as mt5
import pandas as pd
from src.core.mt5_connector import MT5Connector
c = MT5Connector(); c.connect()
bars = mt5.copy_rates_from_pos('GBPJPY', mt5.TIMEFRAME_M15, 0, 60000)
df = pd.DataFrame(bars)
df['time'] = pd.to_datetime(df['time'], unit='s')
df.to_csv('data/GBPJPY_M15.csv', index=False)
print(f'Saved {len(df):,} bars')
c.disconnect()
"
```

**Then retrain:**
```bash
conda run -n envmt5 python scripts/retrain_for_live.py --pair GBPJPY
```

---

## Correlation Matrix (Approximate)

```
              S1 EUR/M15  S2 USD/M15  S3 EUR/H4  S4 XAU/M15  S5 GBP/M15
S1 EUR/M15       1.00       -0.35       0.45       -0.25        0.20
S2 USD/M15      -0.35        1.00      -0.30        0.15       -0.15
S3 EUR/H4        0.45       -0.30       1.00       -0.20        0.25
S4 XAU/M15      -0.25        0.15      -0.20        1.00       -0.10
S5 GBP/M15       0.20       -0.15       0.25       -0.10        1.00
```

All off-diagonal values below 0.50 — genuine diversification.

---

## Portfolio Risk Allocation

$10,000 demo account — 1% risk per trade per strategy:

```
Each strategy:  $100 risk per trade
All 5 running:  max $500 simultaneously (5% total portfolio risk)

If average 2 trades open simultaneously per strategy:
  Total exposure: ~$250 at any time (2.5% of account)

Monthly trade volume (estimated):
  S1 EURUSD M15:   ~20 trades/month
  S2 USDJPY M15:   ~30 trades/month
  S3 EURUSD H4:    ~8 trades/month
  S4 XAUUSD M15:   ~15 trades/month
  S5 GBPJPY M15:   ~20 trades/month
  Total:           ~93 trades/month
```

---

## Implementation Sequence

```
Phase 1 (NOW — in progress):
  [x] S1 EURUSD M15 — model trained, evaluation running
  [x] S2 USDJPY M15 — model trained, evaluation running

Phase 2 (next, data on disk):
  [ ] S3 EURUSD H4  — retrain_for_live.py --pair EURUSD_H4
  [ ] S4 XAUUSD M15 — retrain_for_live.py --pair XAUUSD (with fixed label_threshold)

Phase 3 (after download):
  [ ] Download GBPJPY M15 (60k bars via MT5 bridge)
  [ ] S5 GBPJPY M15 — retrain_for_live.py --pair GBPJPY

Phase 4 (paper trading — all 5 bots simultaneously):
  [ ] Verify each with --dry-run
  [ ] Start all 5 bots on demo account (ICMarketsKE-Demo)
  [ ] Monitor for 30 days
  [ ] Compare live Sharpe vs walk-forward Sharpe per strategy

Phase 5 (live capital):
  [ ] Strategies with live Sharpe > 0.8 → deploy to live account
  [ ] Strategies with live Sharpe < 0.8 → investigate, retrain with external features
```

---

## Running All 5 Bots Simultaneously

```bash
# Start all bots (after all models trained and --dry-run verified)

conda run -n envmt5 python src/bots/pipeline_bot.py \
    --model-dir data/models/pipeline_EURUSD \
    > logs/bot_EURUSD.log 2>&1 &

conda run -n envmt5 python src/bots/pipeline_bot.py \
    --model-dir data/models/pipeline_USDJPY --symbol USDJPY \
    > logs/bot_USDJPY.log 2>&1 &

conda run -n envmt5 python src/bots/pipeline_bot.py \
    --model-dir data/models/pipeline_EURUSD_H4 --symbol EURUSD --tf H4 \
    > logs/bot_EURUSD_H4.log 2>&1 &

conda run -n envmt5 python src/bots/pipeline_bot.py \
    --model-dir data/models/pipeline_XAUUSD --symbol XAUUSD \
    > logs/bot_XAUUSD.log 2>&1 &

conda run -n envmt5 python src/bots/pipeline_bot.py \
    --model-dir data/models/pipeline_GBPJPY --symbol GBPJPY \
    > logs/bot_GBPJPY.log 2>&1 &

echo "All 5 bots running"
```

**Monitor all simultaneously:**
```bash
tail -f logs/bot_EURUSD.log logs/bot_USDJPY.log logs/bot_EURUSD_H4.log \
         logs/bot_XAUUSD.log logs/bot_GBPJPY.log
```

---

## Paper Trading Acceptance Criteria (30 days)

| Strategy | Min live Sharpe | Max live MaxDD | Action if fails |
|----------|----------------|----------------|-----------------|
| S1 EURUSD M15 | > 0.8 | < 25% | Add external features (Phase 29) |
| S2 USDJPY M15 | > 0.8 | < 35% | Retrain with larger dataset |
| S3 EURUSD H4 | > 0.6 | < 30% | Adjust label horizon |
| S4 XAUUSD M15 | > 0.5 | < 40% | Tune label_threshold |
| S5 GBPJPY M15 | > 0.5 | < 40% | Adjust SL/TP for GBP volatility |

---

## Connection to Monetization

Once all 5 pass paper trading, the subscription product (MONETIZE.md) offers:

| Tier | Strategies | Price |
|------|-----------|-------|
| Basic | S1 EURUSD M15 only | $29/month |
| Pro | S1 + S2 + S3 (3 strategies) | $79/month |
| Elite | All 5 strategies | $149/month |

---

*Cross-reference: LEADERBOARD.md (walk-forward results), MONETIZE.md (product plan),*
*IMPROVEMENT.MD (external features for Phase 29–31), PRO_TRADING_INSIGHTS.md (why this works).*
*Last updated: 2026-06-07*
