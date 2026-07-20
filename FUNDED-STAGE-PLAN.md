# FUNDED (MASTER) STAGE PLAN — do this BEFORE the first funded trade

Trigger: a challenge account passes both phases and becomes a **Master / funded**
account. Funded rules differ from Evaluation, and one of them **breaks the current
book**. Do not just carry the Evaluation config over.

Measured 2026-07-20 with `scripts/v5_holding_constraints.py`. Related:
`FUNDINGPIPS-READY.md` (go-live), `V5_FINDINGS.md` (settled experiments).

---

## 1. What changes at Master

| Rule | Evaluation (now) | Master (funded) |
|---|---|---|
| Weekend holding | **ALLOWED** | **NOT allowed** (FundingPips 2-Step Flex, since 29-Jan-2026). Trades auto-closed before Friday close. NOT a hard breach. |
| Overnight holding | ALLOWED | Allowed at FundingPips/FTMO today — **but if any firm bans it, see §4** |
| News | Holding through news OK; *purposely trading* news prohibited | Profits from trades opened/closed within **±5 min** of high-impact (red-folder) news **do not count**, unless executed ≥5 h before |
| Consistency | none (2-Step) | profit-concentration policy may require 4 profitable days if one idea > 60% of the target (accounts created on/after 27-Jun-2026) |

FTMO funded has its own variants (weekend auto-close, ±5 min news window,
leverage caps Crypto 1:2 / Indices 1:20 / Metals 1:30 — non-binding at our ~0.3x gross).

---

## 2. Measured impact of the holding bans

Book = GOLD + ETH + DJI (D1 proxies), FundingPips Flex rules, eval 2017+:

| Scenario | Sharpe | CAGR | maxDD | Pass % | Median |
|---|---:|---:|---:|---:|---:|
| Hold through everything (Evaluation) | **+1.14** | 7.7% | −9.4% | **98.0%** | 20.8mo |
| **No weekend holding** | +0.71 | 4.5% | −11.1% | **68.2%** | 25.0mo |
| **No overnight holding** | **−1.03** | −4.1% | −33.6% | **0.4%** | — |

---

## 3. THE FIX for the weekend ban — drop the crypto sleeve

| Book under weekend ban | Sharpe | Pass % |
|---|---:|---:|
| GOLD + ETH + DJI (Evaluation book) | +0.71 | 68.2% |
| **GOLD + DJI (drop crypto)** | **+1.08** | **79.1%** |
| GOLD + DJI + SPX | +1.05 | 77.9% |
| GOLD + DJI + NDX | +1.12 | 67.0% (worse pass — too volatile for the daily line) |
| GOLD only | +0.73 | 57.4% |

**Why crypto specifically:** crypto trades **24/7**. Forced flat Friday→Monday, ETH
keeps trending without you — you miss real moves. Gold and indices are *closed* over
the weekend anyway, so being flat costs only the Monday opening gap, which is far
smaller. Crypto is the one sleeve uniquely punished by a weekend rule.

**Decision: at Master, swap the crypto sleeve for a second index.**
Preferred: **GOLD + DJI** (best pass) or **GOLD + DJI + SPX** (slightly more
diversified, near-identical). Avoid NDX as the third — higher Sharpe but it breaches
the daily line more often.

---

## 4. If a firm bans OVERNIGHT holding — DO NOT TRADE THIS BOOK THERE

Sharpe **+1.14 → −1.03**. Not a tuning problem: for indices and gold essentially all
the long-run drift happens **overnight** (close→open), so being flat every night hands
away the exact thing the strategy harvests, while paying ~500 crossings/yr instead of
~12. No dial fixes this. An intraday replacement is also ruled out — see V5_FINDINGS
§3c (fast/intraday trend dies on costs).

---

## 5. Implementation checklist (when funded)

1. **Config**: copy the live config, change `classes` to drop crypto:
   ```json
   "classes": { "xau": ["XAUCHAMP"], "eq_us": ["DJI"] }
   ```
   (or add `"eq_us": ["DJI","SPX"]` for the 3-sleeve variant), and remove the crypto
   entry from `symbols`. New `magic` so funded positions are isolated from any
   challenge account still running.
2. **BUILD REQUIRED — weekend auto-flat.** The executor currently holds through the
   weekend. Add to `v5_basket_challenge_exec.py`:
   - `weekend_flat: {enabled, flatten_at, reopen_at}` in config,
   - flatten ALL positions at `flatten_at` (e.g. Fri 20:45 server) and block new
     entries until `reopen_at` (Sunday open / Monday),
   - unit-test like `challenge_guards` (synthetic clock, assert flatten + no re-entry).
   Without this the firm auto-closes for us — tolerable (not a breach) but we lose
   control of the exit price.
3. **News filter**: already built (`src/v5/news_filter.py`). Tighten to the funded
   window: block entries ±5 min around red-folder events (we currently use ±30/60).
   Set `close_in_profit` per firm guidance.
4. **Re-verify sizing at the funded account size** — min-lot notionals decide the book
   (see `configs/v5_fp_flex_10k.json` `_sizing_note`). A funded 10K behaves differently
   from a funded 100K.
5. **Re-run the sim** with the funded rules before the first trade:
   `python scripts/v5_holding_constraints.py` and `v5_basket_challenge.py --backtest`.
6. **Lower expectations**: ~79% pass and a slower grind vs 98% in Evaluation. Do not
   raise the vol dial to compensate — the dial IS the risk control.

---

## 6. Reproduce

- `scripts/v5_holding_constraints.py` — baseline vs no-weekend vs no-overnight, and
  the book variants under the weekend ban.
- Rules sources: FundingPips help centre (News Trading & Weekend Holding; 2 Step Flex),
  ftmo.com/en/trading-objectives.

_Last updated 2026-07-20._
