# FUNDINGPIPS-READY — go-live runbook (basket challenge)

Single source of truth for taking the diversified basket bot live on a real
FundingPips account. Everything below is **proven and running on the VPS demo**
(68.183.91.240, instance 1). Going live = swap the demo account for the real one,
fill the real symbol names, reset state. No new infrastructure needed.

Related: CHALLENGEBOT.MD (research), V5_FINDINGS.md (what was tried),
deploy/vps-deploy-log.md (build log), deploy/vps_provision.sh (replica script).

_Last updated 2026-07-18 — switched to the FOCUSED XAU+BTC+NDX book (see §1)._

---

## 1. THE LOCKED STRATEGY (do not change without re-simming)

| Setting | Value | Why |
|---|---|---|
| Model | **2-Step Standard** | 5% daily / 10% max is the most forgiving for a low-vol book; 4%-daily Flex/3%-daily Pro are riskier. |
| Vol dial | **7%** (6% = safer/slower) | 7% → best pass/speed balance; 6% if you want max safety. |
| Book | **FOCUSED: XAU champion + BTC + NDX, equal ⅓** | 3 uncorrelated sleeves (corr 0.04–0.12). Beats the old 12-instrument basket on every FP metric — see §4. Old 6-class kept as `BASKET_FULL` for revert. |
| **Vol-targeting** | **ON** (`VOL_TARGET=True`) | causal trailing-vol × drawdown scaler; +0.17 Sharpe, +2–3 pass pts |
| Weighting | **equal-class (⅓ each)** | Sharpe-weighting was a lookahead illusion — do NOT use |
| Direction | **long-only** | shorts DISPROVEN (drift assets bleed when shorted) |
| Guards | daily flatten −3.5%, halt −8%, targets +8%/+5% | buffers before the firm's 5%/10% |
| Reconcile buffer | 0.15 (raise to **0.25** on a hedging acct to cut churn) | |

**Expected performance (XAU+BTC+NDX, $100K, 7% vol):**
- Eval Sharpe **1.71** (2021+ 1.28) · FundingPips pass **~98.7%** · median **~11.4 months** (p75 ~17mo)
- fail-daily **0.0%** · fail-DD **1.4%** (vs old basket 5.7%) · maxDD −12.3%
- Live weights ≈ NDX 0.41 / BTC 0.30 / XAU 0.28 (equal risk; leverage differs by asset vol).
- Beats the old 12-instrument basket (SR 1.43 / 94.3% / 12.3mo) with **¼ the symbols**.

---

## 2. GO-LIVE STEPS (when the account is purchased)

**Primary path — point instance 1 at the real account (retire the demo):**

1. **Buy 2-Step Standard**, size to comfort ($100K sizes cleanest; fee refunded after payouts).
2. **Log the terminal into the FundingPips account** (VNC once, save account + enable Algo Trading):
   ```
   ssh -i ~/.ssh/vps_basket_ed25519 trader@68.183.91.240
   DISPLAY=:99 x11vnc -display :99 -localhost -rfbport 5900 -passwd 'Vnc4Login!' -forever -bg
   # laptop: ssh -L 5901:localhost:5900 trader@68.183.91.240 ; vncviewer localhost:5901
   # File→Login: <FP login>/<FP server>, SAVE ACCOUNT, enable Algo Trading; then pkill x11vnc
   ```
3. **Map the 3 real symbols** in `configs/v5_basket_challenge.json` → set each `fp_symbol` to the
   FundingPips Market-Watch name: **XAUCHAMP→XAUUSD/GOLD**, **BTC→BTCUSD**, **NDX→US100/NAS100/USTEC**.
   Discover exact names: run the symbol sweep in deploy/vps-deploy-log.md (§symbols).
4. **Reset state** (fresh anchor on the real balance):
   `rm -f data/v5_runs/basket_challenge_live_state.json`
5. **Confirm attach + account**:
   `python -c "from src.core.mt5_connector import get_mt5 as g; m=g('localhost',18812); m.initialize(); print(m.account_info().login, m.account_info().server)"`
6. **Dry-run one pass** (no --execute) → verify targets show real lots, guards action=trade, correct account.
7. **Go live** — the timer already runs `--live --execute`; just confirm it's the FP account. First live
   pass opens the basket. Watch the daily email + guard log.

**Alternative — keep the demo, add a 3rd instance:** mirror the cent-instance recipe
(deploy/vps-deploy-log.md §2nd instance): prefix `~/.mt5c`, display `:101`, bridge **18814**,
`MT5_BRIDGE_PORT=18814`. Only if you want demo + real running simultaneously (needs 8GB RAM).

---

## 3. WHAT'S PROVEN vs WHAT TO VERIFY ON THE REAL ACCOUNT

**Proven (on demo):** connect/attach, guards trip correctly, order execution (FOK), currency-correct
sizing (order_calc_margin), vol-targeting, real-time 60s guard, daily email, reboot survival, 10-sleeve
live basket. **Verify on the real account:** exact symbol names, crypto availability (add BTC/ETH),
netting vs hedging (FP is usually netting → cleaner, no ticket pile-up), account currency = USD
(no conversion needed, unlike the KES demo).

---

## 4. IMPROVEMENTS FOUND (and dead-ends — don't re-litigate)

- ✅ **Vol-targeting** — the one real win (SR 1.26→1.43). Now live in `v5_basket_challenge.py::risk_scalar`.
- ❌ **Long/short per sleeve** — SR 1.26→0.23. Shorting drift assets bleeds. Confirmed dead.
- ❌ **Crash-hedge shorts** — worse (0.97). ❌ **Faster speeds** — no gain.
- ❌ **Sharpe-weighting sleeves** — 1.46 in-sample but 1.25 walk-forward = lookahead illusion.
- Vol dial frontier: 5%→97%/21mo · 6%→95%/17mo · 7%→92%/14mo · 8%→82%/11mo.
Detail: `data/v5_runs/basket-ls-experiment/REPORT.md`.

---

## 5. PROTECTION & MONITORING (already live)

- **Real-time guard** — `xau-basket-guard.timer` every 60s: flatten if equity breaches −3.5% daily / −8% overall.
- **Reconcile** — hourly (`xau-basket-dry.timer` → live wrapper).
- **Daily email** — `xau-basket-report.timer` 20:55 UTC → kipngenol@gmail.com (gains + rule-adherence %).
  Needs the Gmail app password in `~/MT5/.env.mail` (already sending on demo).
- Logs: `data/v5_runs/vps-basket-*.log`, journals `basket_challenge_*_log.csv`.

---

## 6. PRE-LIVE HARDENING CHECKLIST (recommended before the real account)

- [ ] **Widen reconcile buffer to 0.25** (`configs/v5_basket_challenge.json`) — cuts spread churn.
- [ ] Consider **vol-target daily re-scale** (not per-pass) to reduce hourly re-trades (code tweak).
- [ ] Confirm **FundingPips is netting** (if hedging, the buffer widen matters more).
- [ ] Map **BTCUSD + ETHUSD** — crypto is the single biggest Sharpe add (+0.20).
- [ ] If running demo + real together, **resize VPS to 8GB**.
- [ ] Funded (Master) stage differs (news ±5min restricted, weekend rules, leverage caps
      Crypto 1:2 / Indices 1:20 / Metals 1:30) — enable the news filter + review before first funded trade.
- [ ] **Rotate the leaked Anthropic API key** (still outstanding).

---

## 7. FALLBACK

If the basket underperforms live or a symbol is missing, the **single-XAU champion**
(`v5_xau_challenge.py`, 88% idealized / 61% realistic pass) is the documented fallback —
but it's weaker on the daily-loss rule. Prefer fixing the basket (map more symbols) over falling back.
