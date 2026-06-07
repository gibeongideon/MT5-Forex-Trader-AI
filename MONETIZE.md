# Monetization Plan — MT5 AI Trading Signal Service

> How to turn the champion model (XGBoost + enc8, +3.13 Sharpe) into a subscription product.

Last updated: 2026-06-07

---

## Three Business Models

### Model 1 — Signal Service (Easiest, Start Here)

You run the model. Users receive signals. They trade manually.

```
Your server runs champion model 24/7
        ↓
Signal fires: "BUY EURUSD @ 1.0842  SL 1.0812  TP 1.0902"
        ↓
Delivered via: Telegram bot / Email / Discord / Webhook
        ↓
User decides to take the trade on their own MT5
```

| | |
|--|--|
| **Pros** | No access to user funds, zero regulation risk, easiest to build |
| **Cons** | Users must trade manually, signal delay matters |
| **Revenue** | $29–$99/month via Stripe |
| **Time to build** | 1–2 weeks |

---

### Model 2 — EA + API (Most Scalable, Recommended)

You host the model as a REST API. Users install a lightweight MQL5 Expert Advisor
on their own MT5. The EA calls your API every bar, receives a signal, and auto-executes
trades on the user's own broker account.

```
User's MT5
  └── Your EA (MQL5, ~150 lines)
        │  every M15 bar: sends OHLCV → POST /signal
        │  receives: {signal: "buy", confidence: 0.71, sl: 30, tp: 60}
        │  executes: trade on user's own broker/account
        ▼
Your Backend (FastAPI on Linux VPS)
  ├── POST /signal  → runs XGBoost+enc8 → returns prediction
  ├── GET  /health  → uptime check
  ├── POST /auth    → validates API key (subscription active?)
  └── Stripe webhook → creates/expires API keys on payment events
```

**Key advantages:**
- Your model code (IP) stays on your server — users never see it
- Users trade on their own broker — you never touch their money
- Scales to 1000+ users with one $20/month VPS
- Each user controls their own lot size and risk settings
- No financial regulation needed (signal service, not managed funds)

| | |
|--|--|
| **Pros** | Fully automated for users, scalable, IP protected |
| **Cons** | Need to build API + MQL5 EA |
| **Revenue** | $29–$149/month tiered |
| **Time to build** | 4–6 weeks |

---

### Model 3 — Copy Trading (Fastest to Launch)

Your live account trades. Subscriber accounts mirror every trade automatically
via MetaQuotes ecosystem or third-party platforms (Duplikium, MyFXBook Autotrade).

```
Your master MT5 account → trade opens
        ↓
Copy trading platform (MetaQuotes / Duplikium / MyFXBook)
        ↓
All subscriber accounts mirror the trade instantly
```

| | |
|--|--|
| **Pros** | Zero coding — platforms handle everything |
| **Cons** | Need real capital in master account, performance fully public |
| **Revenue** | Monthly subscription + optional 20% performance fee |
| **Time to build** | Days (platform does the work) |

---

## Recommended Architecture — Model 2 (EA + API)

This reuses almost all existing code. `PredictorPipeline.predict()` already works
for live inference — the API is a thin wrapper around what we've built.

### Backend Stack

```
VPS (Ubuntu, ~$20/month)
  ├── FastAPI          — signal endpoint, auth middleware
  ├── PostgreSQL       — users, subscriptions, API keys, usage logs
  ├── Redis            — rate limiting per API key (e.g. 1 req/15min)
  ├── Stripe           — payment processing, webhook for key lifecycle
  └── Nginx            — reverse proxy, SSL termination
```

### Existing Code That Gets Reused

| File | How it's reused |
|------|----------------|
| `src/pipeline.py` → `PredictorPipeline.predict()` | Core inference — called by the API on every request |
| `data/models/pipeline/` | Champion artifacts (encoder.pt, scaler.joblib, model.joblib) loaded at API startup |
| `src/core/mt5_connector.py` | Used by your own server bot only — users don't need it |
| `src/bots/pipeline_bot.py` | Refactor into `api/signal_handler.py` |
| `deploy/mt5_bot.service` | Extended to also run the FastAPI server as a service |

### New Code Needed

| File | What it does |
|------|-------------|
| `api/main.py` | FastAPI app — `POST /signal`, `GET /health`, `GET /status` |
| `api/auth.py` | API key validation middleware — checks key exists + subscription active |
| `api/billing.py` | Stripe webhook — creates key on payment, expires on cancellation |
| `api/models.py` | SQLAlchemy models: User, Subscription, APIKey, SignalLog |
| `MT5_Signal_EA.mq5` | MQL5 EA — sends OHLCV to API, receives signal, executes trade |
| `api/admin.py` | Simple dashboard — active subscribers, signal count, revenue (optional) |

### API Endpoints

```
POST /signal
  Headers: X-API-Key: usr_abc123xyz
  Body:    {open, high, low, close, tick_volume, timestamp}[]  (last 200 bars)
  Returns: {signal: "buy"|"hold"|"sell", confidence: 0.71,
            sl_pips: 30, tp_pips: 60, model_version: "v1.0"}

GET /health
  Returns: {status: "ok", model_loaded: true, uptime_seconds: 86400}

POST /auth/validate
  Headers: X-API-Key: usr_abc123xyz
  Returns: {valid: true, plan: "pro", pairs_allowed: ["EURUSD","GBPUSD"],
            requests_today: 42, requests_limit: 96}

POST /webhooks/stripe
  Handles: checkout.session.completed → create API key
           customer.subscription.deleted → expire API key
           invoice.payment_failed → send warning email
```

---

## Subscription Tiers

| Tier | Price | Pairs | Auto-execution | Features |
|------|-------|-------|---------------|----------|
| **Basic** | $29/mo | EURUSD only | Yes (via EA) | Standard signals |
| **Pro** | $79/mo | EURUSD + GBPUSD + USDJPY | Yes | Custom risk %, email alerts |
| **Elite** | $149/mo | All pairs | Yes | Regime filter, priority API, Discord group |

---

## MQL5 EA — How It Works

The EA runs on the user's MT5. On every new M15 bar it:
1. Collects the last 200 OHLCV bars from their MT5
2. Sends them to your API (`POST /signal`) with their API key in the header
3. Receives `{signal, confidence, sl_pips, tp_pips}`
4. If `signal == "buy"` and no open position → opens a buy order
5. Manages SL/TP and breakeven exactly as `pipeline_bot.py` does today

The EA is ~150 lines of MQL5. Users download it from your website after subscribing,
drop it into their MT5 `Experts/` folder, enter their API key in the EA settings, and
attach it to the EURUSD M15 chart. That's the entire user setup.

---

## Legal Considerations

| Issue | How to handle |
|-------|--------------|
| Financial regulation | Signal service only (not managed funds) — most jurisdictions don't require a license |
| Disclaimer | "Past performance does not guarantee future results" — mandatory on all marketing |
| User funds | Never held by you — EA trades on user's own broker account |
| Jurisdiction | UK FCA, US NFA have specific signal service rules — consult a lawyer before launch |
| Terms of service | Clearly state: results not guaranteed, past Sharpe is backtest not live |

---

## Infrastructure Cost vs Revenue

| Users | VPS cost | Stripe fees | Net revenue (Basic $29) |
|-------|----------|-------------|------------------------|
| 10 | $20/mo | ~$9/mo | ~$261/mo |
| 50 | $20/mo | ~$45/mo | ~$1,380/mo |
| 200 | $40/mo | ~$174/mo | ~$5,586/mo |
| 1000 | $80/mo | ~$870/mo | ~$28,050/mo |

One VPS handles thousands of users — the model is loaded once and serves all requests.

---

## Build Timeline

```
Week 1:  FastAPI signal endpoint + API key auth middleware
Week 2:  PostgreSQL schema + Stripe integration + webhook handler
Week 3:  MQL5 EA (signal fetch + order execution + SL/TP management)
Week 4:  End-to-end test with your own MT5 account as first "subscriber"
Month 2: Beta with 5–10 invited users — collect feedback, fix edge cases
Month 3: Public launch — landing page, Stripe checkout, EA download
```

---

## Next Implementation Steps (When Ready)

```bash
# 1. Create the API project structure
mkdir -p api/{routers,models,middleware}

# 2. Install API dependencies
conda run -n envmt5 pip install fastapi uvicorn sqlalchemy psycopg2 stripe redis

# 3. Start API server (development)
conda run -n envmt5 uvicorn api.main:app --reload --port 8000

# 4. Test signal endpoint with your existing champion
curl -X POST http://localhost:8000/signal \
  -H "X-API-Key: test_key" \
  -H "Content-Type: application/json" \
  -d @data/sample_bars.json
```

---

## Phase Dependency

Complete these first before building the product:
- Phase 24: Cross-market features (may improve champion before locking for production)
- Phase 25/26: Regime + meta-label results (incorporate if they improve Sharpe)
- 30-day paper trading validation (mandatory before charging subscribers)
- `scripts/retrain_champion.py` on full dataset → lock model version as `v1.0`
