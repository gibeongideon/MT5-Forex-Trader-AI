# SPRINT TO FUNDED — risk/speed options report (2026-07-20)

Question: can we get funded FASTER by accepting more risk, then restructure to the
safe book once funded? Measured with `scripts/v5_sprint_analysis.py` on the live
10K book (XAU + ETH + DJI, eval Sharpe 1.41), 4,000 bootstrap paths, day_safety 1.5.

**Why sprinting is rational at all:** a challenge is an ASYMMETRIC bet — downside is
capped at the entry fee (~$60), upside is a funded account. So the objective is not
risk-adjusted return, it is *cheapest/fastest expected route to funded*, where failing
is survivable and retryable.

**Benchmark:** a ZERO-EDGE random walk between +10% and −12% hits the target first
54.5% of the time (= maxloss/(target+maxloss)). Any pass rate near that means the
edge has stopped mattering and you are coin-flipping.

---

## 1. THE BINDING CONSTRAINT IS THE DAILY LIMIT — NOT THE MAX LOSS

Sprinting on **Flex** (your current account, −4% daily):

| vol dial | pass% | fail-DAILY% | fail-DD% | median mo | attempts | E[fees] | E[months] | risk |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 7% (live) | **98.9** | 0.0 | 1.1 | 17.4 | 1.01 | $61 | 17.5 | **1 – LOW** |
| 10% | 58.4 | **39.4** | 2.2 | 9.0 | 1.71 | $103 | 11.1 | 7 – VERY HIGH |
| 14% | 46.4 | **50.5** | 3.0 | 5.0 | 2.16 | $129 | 6.9 | 8 – VERY HIGH |
| 18% | 24.4 | **73.8** | 1.8 | 2.5 | 4.10 | $246 | 5.1 | 9 – EXTREME |
| 25% | 16.5 | **83.1** | 0.4 | 1.2 | 6.06 | $364 | 3.2 | 10 – EXTREME |
| 45% | 11.4 | 88.6 | 0.0 | 0.5 | 8.77 | $526 | 1.8 | 10 – EXTREME |

Read the failure columns: **fail-DD stays ~0–3% everywhere.** You almost never bleed
down to the −12% max loss. You get knocked out by ONE BAD DAY. On Flex's −4% daily
line, any dial above ~8% is essentially a coin flip decided by a single session.

---

## 2. THE PRODUCT MATTERS MORE THAN THE DIAL

Same book, same dial, different challenge model:

| dial | FLEX (−4% daily) | STANDARD (−5% daily) | FTMO (−5% daily) |
|---:|---:|---:|---:|
| 7% | 98.9% / 17.4mo | 97.4% / 13.6mo | 97.3% / 16.1mo |
| **10%** | **58.4%** | **91.3%** | **91.1%** |
| 14% | 46.4% | 62.5% | 59.7% |
| 18% | 24.4% | 48.1% | 44.3% |
| 25% | 16.5% | 33.4% | 29.1% |

**One extra point of daily headroom (4%→5%) buys +33 points of pass rate at 10% vol**
(58% → 91%), because fail-daily collapses from 39.4% to 0.0%. Standard also has a
lower P1 target (+8% vs +10%), which compounds the advantage.

---

## 3. RECOMMENDED OPTIONS (risk scale 1–10)

| # | Plan | Pass% | E[time to funded] | E[fees] | Risk |
|---|---|---:|---:|---:|---|
| **A** | **Flex @ 7% — current live setup** | **98.9** | 17.5 mo | $61 | **1 – LOW** |
| **B** | **Standard @ 10% — "smart sprint"** | **91.3** | **9.1 mo** | $66 | **3 – MODERATE** |
| C | Standard @ 14% | 62.5 | 5.5 mo | $96 | 6 – HIGH |
| D | Standard @ 18% | 48.1 | 3.8 mo | $125 | 8 – VERY HIGH |
| E | Flex @ 14% (sprint on the account you own) | 46.4 | 6.9 mo | $129 | 8 – VERY HIGH |
| F | Any product ≥25% vol | ≤33 | 2–3 mo | $180–520 | 10 – EXTREME |

**Best value = Option B.** Standard @ 10% vol gets ~91% pass in ~9 months — roughly
**half the time of the current plan at almost the same safety**, for one extra fee.
It dominates Flex @ 7% on a time-adjusted basis. The "real" sprints (C/D) buy months
at a steep price in pass probability.

**Beyond ~18% vol it is value-destroying**: pass collapses toward the coin-flip line,
fees multiply, and the time saved flattens out because you keep re-entering.

---

## 4. RISKS THE SIMULATION DOES **NOT** CAPTURE (read before sprinting)

1. **Our own guard becomes the binding constraint.** The bot flattens at −3.0% daily
   (Flex) / −3.5% (FTMO). At 14%+ vol a 3% down-day is routine, so the guard would
   flatten and lock out almost daily — the bot would spend most of its life flat.
   `fp_sim` models the FIRM's limits, **not our flatten-and-lock**, so it
   **OVERSTATES** high-vol performance. Sprinting requires loosening the guard, which
   removes the very buffer that protects the account.
2. **Edge decay.** These runs assume the Sharpe-1.41 book keeps working. At 7% vol a
   weak patch is survivable; at 18% it ends the account. High vol converts "unlucky
   month" into "dead account".
3. **Fat tails / gaps.** Block-bootstrap resampling of history understates gap risk.
   A weekend gap can jump straight through both our guard and the firm's daily line.
4. **Leverage/margin and execution.** 18%+ vol means ~3–4x the current gross exposure;
   slippage and margin behaviour are not modelled.
5. **Reputational/account limits.** Repeated fast failures may attract firm scrutiny;
   some firms restrict serial re-purchases.

---

## 5. VERDICT

- The **sprint is real but modest**: you can compress ~17 months to ~9 at
  near-equal safety (Option B) — but that comes from **choosing a 5%-daily product**,
  not from cranking the dial.
- **Cranking the dial on Flex is the worst of both worlds** (Option E): the −4% daily
  line means you pay full sprint risk for below-average odds.
- If you sprint, **buy a Standard (or FTMO) account for it** and keep the Flex 10K on
  the safe 7% dial as the reliable path.
- **Do not exceed ~14% vol.** Past that you are paying real money to coin-flip, and
  our own guard invalidates the model anyway.

Restructuring after funding is already documented in `FUNDED-STAGE-PLAN.md`
(weekend-holding ban → drop the crypto sleeve).

---

## 6. BUILD CHECKLIST — when we decide to run the sprint (NOT built yet)

Target = **Option B: FundingPips 2-Step STANDARD @ 10% vol**, run in PARALLEL with the
existing Flex 10K (which stays on the safe 7% dial). Nothing below exists yet.

1. **Buy** a 2-Step Standard account (size to comfort; rules identical across sizes).
2. **Engine — add a sprint model.** `scripts/v5_basket_challenge.py::MODELS` currently
   has `standard` at `vol=0.07`. Add:
   ```python
   "standard_sprint": dict(vol=0.10, p1=0.08, p2=0.05, daily=0.05, maxloss=0.10,
                           guard_frac=0.035, halt_frac=0.08),
   ```
   (Standard limits, sprint dial. Do NOT raise `vol` on the existing `standard` entry —
   other configs reference it.)
3. **Config** `configs/v5_fp_standard_sprint.json`: `model: "standard_sprint"`,
   **new magic (e.g. 360563)**, own `run_id`, `reset_tz: "UTC+3"`,
   `guards: {daily_guard_frac 0.035, overall_halt_frac 0.08, phase_targets {1:0.08, 2:0.05}}`,
   `classes` = the 10K book `{xau:[XAUCHAMP], crypto:[ETH], eq_us:[DJI]}`.
4. **Re-verify sizing on the REAL account before trading.** Min-lot notionals decide the
   book (see `configs/v5_fp_flex_10k.json` `_sizing_note`). At 10% vol targets are ~43%
   larger than at 7%, which *helps* granularity — but confirm on the actual terminal
   (FundingPips contracts differ from FTMO: SPX500=50, NDX100=20, ETHUSD=1, XAUUSDmicro=10).
5. **VPS instance 4**: prefix `~/.mt5d`, display `:102`, bridge **18815**. Follow
   `vps-three-bot-architecture` memory — clone a prefix, **install a full 598KB
   `servers.dat`** (a pruned one silently prevents login), strip `accounts.dat`, write
   `mt5_login.ini`, prefix-scoped start script. RAM headroom is fine (~310MB/instance,
   ~2.4GB free with three running).
6. **Wrappers/units**: `deploy/sprint_{challenge,guard,report}_cron.sh` +
   `sprint-{challenge,guard,report}.{service,timer}`. **Stagger the reconcile to `:37`**
   (taken: fp10k `:07`, cent `:16`, ftmo `:22`; guards on `:00/2` and `:01/2` — use `:03/2`).
   Include the readiness gate (bridge + real bids) — without it the bot silently no-ops.
7. **Report**: reuse `scripts/challenge_daily_report.py --config ... --state ... --port 18815`
   (already config-driven; no new script needed).
8. **Dry-run first**, then one manual `--live --execute` pass, then enable timers —
   same sequence used for the Flex 10K.

**Stop rules for the sprint account:** if it fails, re-buy at most **twice** (E[fees]
assumes ~1.1 attempts at 10% vol; three failures means the edge or the assumptions are
wrong, not bad luck). **Never raise the dial mid-challenge** and never take the sprint
above 14% — past that our own −3.5% guard invalidates the model (§4.1).

_Reproduce: `python scripts/v5_sprint_analysis.py`._
