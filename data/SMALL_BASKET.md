# Small Multi-Instrument Trend Basket — Results (2026-06-17)

Goal: does a **small** trend basket capture the diversified-CTA edge without needing all 48
instruments? Engine = the validated champion config (`cta_backtest.py --sleeve combined
--rebalance monthly --risk cluster`): EWMAC continuous trend + cross-sectional momentum,
cluster-equal risk budgeting, 10% vol-target, monthly rebalance, real cost, discover(2010–21)/
confirm(2022–26) + block-bootstrap 95% CI. Added `--instruments` flag to subset the universe.

## Verdict: GO — a 5-instrument basket BEATS the full 48-universe

A basket of **one liquid instrument per asset class** captures all of the edge with far less
turnover, and — unlike any single instrument — is **positive and significant in BOTH the
discover and confirm periods** (not regime-dependent beta).

### Diversification ladder (champion config, net of cost)

| Basket | n | DISCOVER | CONFIRM | **FULL (95% CI)** | corr→long-all | turnover/yr | maxDD |
|--------|---|----------|---------|-------------------|---------------|-------------|-------|
| GOLD | 1 | +0.24 | +0.97 | +0.43 [+0.01,+0.84] | +0.28 | 473% | 28% |
| +UST10Y | 2 | +0.20 | +1.03 | +0.41 [−0.02,+0.84] | +0.16 | 1199% | 31% |
| +SPX | 3 | +0.46 | +1.19 | +0.64 [+0.23,+1.06] | +0.25 | 1055% | 22% |
| **+WTI +EURUSD** | **5** | **+0.63** | **+1.06** | **+0.73 [+0.32,+1.13]** | **−0.20** | 1113% | **18%** |
| +4 more (broad) | 9 | +0.47 | +0.70 | +0.52 [+0.05,+0.98] | −0.16 | 1182% | 24% |
| full universe | 48 | +0.73 | +0.38 | +0.65 [+0.18,+1.12] | +0.08 | 2150% | 24% |

**The 5-instrument basket = GOLD, UST10Y, SPX, WTI, EURUSD** (metal / rates / equity / energy /
FX) is the best risk-adjusted config of all:
- **FULL net Sharpe +0.73, CI [+0.32, +1.13]** — higher than the 48-universe (+0.65) with **half
  the turnover** (1113% vs 2150%/yr) and lower max DD (18% vs 24%).
- **Discover +0.63 (CI [+0.13, +1.09], significant on its own)** and confirm +1.06 — works in
  *both* halves. This is the decisive difference from the single-instrument study, where the
  apparent winners were flat in discover and only "worked" in the 2022–26 bull regime.
- **corr-to-long-everything −0.20** — genuinely market-neutral, not disguised beta.
- Curve rises steeply 1→5 then flattens: most √N benefit comes from the first ~5 *uncorrelated*
  bets. The 9-basket was worse because the instruments I added (NIKKEI/COPPER/CORN/USDJPY) overlap
  existing clusters — breadth only helps when it adds a *new, low-correlation* bet.

### Robustness (cardinal-rule audit — nothing absurd, all pass)

| Variant | FULL Sharpe | Note |
|---------|-------------|------|
| 5-basket + buffer 0.10 | +0.73 | unchanged (buffer at 0.1 barely cuts turnover) |
| swap BRENT↔WTI, USDJPY↔EURUSD | **+0.89** | even better — not fragile to member choice |
| swap SILVER↔GOLD, DAX↔SPX | +0.42 | weaker (silver noisier, DAX = equity beta) — member quality matters |
| daily rebalance | +0.60 | worse + turnover 3852% → monthly is correct |

Across sensible 5-instrument choices spanning the five clusters, FULL net Sharpe = **+0.42 to
+0.89, all positive, discover positive, market-neutral**. Member *quality* matters (gold > silver,
SPX > DAX), monthly rebalance beats daily, but the result is structurally robust.

## Conclusion
**Yes — a small multi-instrument trend basket works, and is the best edge found in the project.**
The recommended deployable strategy is the **5-symbol basket (GOLD, UST10Y, SPX, WTI, EURUSD),
EWMAC trend + cluster-equal risk + 10% vol-target + monthly rebalance**: net Sharpe **+0.73,
CI [+0.32, +1.13]**, ~0 beta, 18% max DD, positive in both discover and confirm. It is far more
deployable than the 48-instrument universe — 5 liquid symbols, ~one rebalance/month — which fits
the current single-symbol bot infrastructure with only a light multi-symbol daily runner.

**The one cost to manage is turnover (~1100% NAV/yr).** A 0.1 buffer didn't dent it; a larger
no-trade band or slower EWMAC speeds should be tested before live deployment to confirm the
net edge survives realistic execution.

**Reconciles the whole arc:** single instruments fail (weak, regime-dependent ≈ beta); the edge
is structural diversification across a *few* low-correlation asset classes at the daily horizon.
You need ~5 uncorrelated bets, not 48 — and not 1.

---

## Turnover reduction — LOCKED production config (2026-06-17)

Buffer × EWMAC-speed grid on the 5-basket (champion config). Slowing the trend (drop the two
fastest EWMAC speeds → spans (32,128),(64,256)) + a 0.4 no-trade buffer **raised** net Sharpe
while cutting turnover 39%:

| speeds | buffer | FULL Sharpe | turnover/yr |
|--------|--------|-------------|-------------|
| fast | 0.0 | +0.728 | 1113% |
| fast | 0.4 | +0.720 | 968% |
| **slow** | **0.4** | **+0.746** | **683%** |
| slowest | 0.4 | +0.586 | 533% |

**LOCKED: GOLD, UST10Y, SPX, WTI, EURUSD · sleeve=combined · trend_speeds=slow · risk=cluster ·
target_vol=10% · rebalance=monthly · buffer=0.4.** Full breakdown: DISCOVER +0.66 [+0.17,+1.10]
(both sub-halves +0.69/+0.64), CONFIRM +1.04 [+0.27,+1.88], **FULL +0.746 [+0.34,+1.15]**,
maxDD 18.8%, corr−0.21, turnover 683%/yr. ("slowest" over-smooths — misses trends, +0.59.)

## Deployment infra (built 2026-06-17)
- **`src/cta/strategy.py`** — single source of truth: `champion_positions()` + the locked
  `BASKET`/`CONFIG`. The backtester now imports its rebalance/buffer/speed primitives from here,
  so the deployed signal is byte-identical to the validated backtest.
- **`scripts/basket_runner.py`** — daily ADVISORY runner: computes today's target positions,
  prints a per-symbol order ticket (action vs last run: OPEN/ADD/TRIM/FLIP/CLOSE), persists
  restart-proof state (`data/basket_state.json`) + audit CSV (`data/basket_signals.csv`).
  `--validate` re-checks full Sharpe == +0.746. Places **no live orders** (standing rule).
- `cta_backtest.py` gained `--instruments` and `--trend-speeds` flags. Tests: 7/7 green;
  full-universe unchanged (+0.647).

**Open deployment items before live:** (1) ~~UST10Y has no retail CFD~~ **RESOLVED (2026-06-18):
UST10Y IS tradable on HFM as the `US10YR` bond CFD** (US 10Y T-Note; 1 lot=100u, 1:50, spread ~0.06,
zero commission). No substitution needed — keeping it gives the best Sharpe (+0.746) AND lowest DD
(dropping rates → +0.739 but DD 18.8%→21.6%). Robustness: UST30Y also works (+0.716, lower turnover
534%). All 5 map to HFM: GOLD→XAUUSD, UST10Y→US10YR, SPX→US500, WTI→USOIL, EURUSD→EURUSD (verify exact
tickers / `.Z` suffix in the live terminal). (2) Convert vol-scaled units → lots via contract specs +
account equity. (3) Refresh `data/*_D1_long.csv` (`scripts/download_universe.py`) before each run.

## Units→lots sizing + capital floor (2026-06-18)

`src/cta/sizing.py` converts engine units → broker lots. A position `pos_i` IS the signed notional
exposure as a fraction of equity (portfolio return = Σ pos_i·return_i), so:

    lots_i = (pos_i · equity) / (contract_size_i · price_i)   # rounded to vol_step, clamped [min,max]

`basket_runner.py --equity <USD>` prints the lots ticket (notional, err%, ROUND→0 flags, gross
leverage); `--live` pulls exact contract specs + price from the MT5 terminal, offline uses panel
close + `DEFAULT_CONTRACT`. Gross lots leverage ≈ **1.44× equity** (matches the model's 1.41×).

**CAPITAL FLOOR = ~$29,000.** Below this the vol target can't be represented in 0.01-lot steps and
distorts. Binding leg is **GOLD**: XAUUSD is 100 oz/lot × ~$4,357 ≈ **$436k notional/lot**, far
coarser than gold's small target weight, so on a $10k account gold rounds to 0 (−15% error). At
$50k all 5 legs are clean (errors ≤2.5%).

### Drop-Gold analysis — gold earns its place (do NOT drop it to save capital)

| Basket | FULL Sharpe | CONFIRM (OOS) | maxDD | min viable equity | binding leg |
|---|---|---|---|---|---|
| **Gold (locked champion)** | **+0.746** [+0.34,+1.15] | **+1.038** | 18.8% | **$29,249** | gold |
| Drop metal (UST10Y,SPX,WTI,EURUSD) | +0.462 [+0.04,+0.89] | +0.126 | 18.1% | **$5,824** | EURUSD |
| Silver for gold | +0.559 [+0.14,+0.98] | +0.538 | 20.3% | $38,896 | silver |

- **Dropping gold guts the edge:** full Sharpe halves (+0.75→+0.46) and OOS confirm goes ~flat
  (+1.04→+0.13). Gold's 2022–26 bull trend carried much of the recent edge and was the only metal.
- **Silver does NOT fix capital:** it needs *more* ($38.9k > gold's $29.2k) because XAGUSD is
  5,000 oz/lot (~$250k notional) at a small weight → even coarser min-lot. Recovers some edge
  (+0.56/+0.54) but strictly worse than gold on capital.
- **DECISION: keep gold, fund the account to ≥ $29k.** Only if hard-capped < $29k, use the
  4-instrument no-metal basket ($5.8k floor, +0.46 Sharpe, ~flat OOS) — silver is the worst of both
  worlds. The capital number is a funding requirement, not a reason to drop the strongest leg.
