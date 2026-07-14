# CHALLENGEBOT — plan to pass a funded-account challenge with the XAU champion

Status: PLAN (drafted 2026-07-14, from live simulation session).
Owner strategy: long-only conc^1.5 champion (see `configs/v5_xau_dual.json` bot
`champ`, signal `src/v5/xau_dual_signals.py::champion_signal`, research
`data/v5_runs/XAU_SHARPE1_RESEARCH.md`).

## 1. The challenge rules (as quoted)

| Feature          | 1-Step        | 2-Step Standard              |
|------------------|---------------|------------------------------|
| Phases           | 1             | Phase 1 + Phase 2            |
| Profit target    | 10%           | 8% (P1) + 5% (P2)            |
| Daily loss limit | 3% (strict)   | 5%                           |
| Max overall loss | 6% (tight)    | 10%                          |
| Time limit       | Unlimited     | Unlimited                    |
| Min trading days | 3             | 3 per phase                  |

## 2. Feasibility verdict (simulated 2026-07-14)

Method: stationary block bootstrap (20-day blocks, 4,000 paths/config) of the
champion's ACTUAL daily net returns (continuous engine, buffer 0.1, half-spread
+ $0.10 costs), rules applied path-by-path. Pessimistic variant re-runs the sim
on the 2016-2020 weak-years stream only (Sharpe ~0.8, pre-bull).

Core insight: **time is unlimited, so risk is the only enemy** — run the bot at
a fraction of deployed risk and let a Sharpe~1 edge grind to the target.

Pass rates (weak-years stream = conservative):

| Model  | Ann. vol target | Pass % | Median time | Dominant failure    |
|--------|-----------------|--------|-------------|---------------------|
| 2-step | 5%              | 91%    | ~22 mo      | max-loss (9%)       |
| 2-step | 8%              | 81%    | ~12 mo      | max-loss (19%)      |
| 1-step | 4%              | 89%    | ~19 mo      | max-loss (11%)      |
| 1-step | 5%              | 75%    | ~12 mo      | daily 3% (14%!)     |
| 1-step | 10% (deployed)  | ~59%   | ~4 mo       | daily 3% (24%)      |

**DECISION: take the 2-STEP model, run at ~5% ann vol (half deployed risk);
optionally 8% vol if speed > certainty.** The 1-step's 3% daily rail is a
single-bad-day killer for gold (worst historical day −6.4% at deployed risk).

## 3. Fine print to verify BEFORE paying the fee (any one flips the verdict)

- [ ] Max overall loss STATIC on initial balance (assumed) — if TRAILING
      high-water mark, re-simulate; difficulty jumps sharply.
- [ ] Weekend/overnight holding ALLOWED — mandatory; H4 trend holds for days.
      If banned, this bot cannot run there. Also check news-trading bans.
- [ ] Daily loss measured on floating intraday equity (assumed close-to-close
      in sim — intraday runs deeper; extra reason to avoid the 1-step).
- [ ] Consistency rule (e.g. "no day > X% of total profit") — trend P&L is
      lumpy; a few big days carry the year.
- [ ] Minimum activity rule vs our ~3-4 trades/month.
- [ ] Metals (XAUUSD) tradable, leverage ≥ 1:20, spread ≤ ~$0.35 all-in.

## 4. Build plan

Base: fork of the proven executor `scripts/v5_xau_dual.py` (same engine,
signal, reconcile loop, --live/--execute double lock) → `scripts/v5_xau_challenge.py`
+ `configs/v5_xau_challenge.json`. New requirements beyond the dual bot:

1. **Risk dial**: `risk_frac` 0.01 → **0.005** (≈5% ann vol; 0.008 for the 8%
   variant). Keep conf_risk_scale {0.5, 1.0, 1.5}, 3×ATR SL/trail.
2. **Hard equity guards (the bot must fail-safe BEFORE the firm's rails):**
   - Daily guard: at each pass, if equity − day_start_equity ≤ −2.0% (buffer
     under the 5% rule): close position, no new entries until next day.
   - Overall guard: if equity ≤ initial × (1 − 6%) (buffer under 10%): flatten
     and HALT permanently (challenge is a write-off at that point anyway —
     protect the retry fee by stopping early).
   - Persist `day_start_equity`/`initial_balance` in a state JSON, not memory.
3. **Phase tracker**: state file records phase (1/2), phase-start balance,
   target; on target hit → log loudly + notify (phase transitions are manual
   at most firms — new credentials for Phase 2).
4. **Separate MT5 login**: challenge account gets its own terminal profile /
   `.env`; NEVER share the terminal with the cent-account bots (different
   broker anyway). New systemd user units `xau-challenge.{service,timer}`,
   hourly like xau-dual.
5. **Min-lot check**: challenge accounts are typically $10k-$200k standard
   contracts (100 oz) — at 0.5% risk over 3×ATR this sizes fine at 0.01+ lots
   for ≥$10k; verify with the firm's contract spec on day 1 (dry pass).
6. **Journal**: same TradeJournal db, bot=v5_xau_challenge; nightly equity row
   for guard-audit.

## 5. Rollout

1. Verify §3 fine print against the firm's actual T&Cs. Re-simulate if any
   assumption breaks (sim harness: session of 2026-07-14, reproducible from
   `data/v5_runs/xau-sharpe1-lab/xau_lab.py` + champion recipe).
2. Build executor + config + guards; unit-test the guards with synthetic
   equity paths (guard must trip in the SAME pass that breaches the buffer).
3. Dry-run (no --execute) against the challenge account for 2-3 days.
4. Go live at 5% vol. Expect a LONG campaign (median ~22 mo; ~12 mo at 8%).
5. On funded: same bot, same risk (funded accounts usually keep the same
   rails; payout > speed).

## 6. Expectations honesty

- ~90% is not 100%: ~1 in 10 challenges fails even played correctly — budget
  for a retry fee.
- The edge is half gold-drift: a multi-year gold bear compresses the pass
  probability toward the weak-years numbers (that is why they are quoted).
- Do NOT be tempted to raise risk after a good month — the sim says the low
  dial IS the strategy.
