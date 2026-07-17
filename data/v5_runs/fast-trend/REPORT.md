# Fast / intraday TREND runner — research report (2026-07-17)

**Question.** Build a *short-term* trend runner to complement the slow H4
long-only champion (acct 360542): higher turnover, opens/closes more trades
per day. Is there a fast-trend edge on XAUUSD net of the real cent-account
$0.34 spread?

**Method.** Two engines, eval 2017+, Sharpe = daily-resampled × √252.
- `scripts/v5_xau_fast_trend_lab.py` — vectorized vol-targeted CTA sweep
  (EWMAC / breakout / session-ORB / intraday-momentum × M15/M30/H1 ×
  long-only vs long-short), with a Carver no-trade buffer, cost = half-spread
  per position change ($0.34 round-trip), correlation & combined-book Sharpe
  vs the champion.
- `scripts/v5_xau_fast_trend_discrete.py` — the REAL lot/stop/%-risk engine
  (`src.v5.xau_trend.run_trades`) driven by a monkeypatched fast champion
  signal; spread forced to $0.34; this is the deployable number.

## Findings

1. **Long-only dominates long-short at every speed** — same "kill the shorts"
   lever as the H4 champion. Gold up-drift + shorts bleed. (LS net SR ~0.4–0.6
   vs LO ~0.8–1.0 in the vectorized lab.)

2. **The trend edge is gross-positive at ALL speeds** (breakout gross SR up to
   1.3) but **net Sharpe falls as turnover rises** — spread is the tax. There
   is **no intraday-specific alpha**: session-ORB and intraday-momentum are
   gross-positive but **net-DEAD** (same failure mode as the dead fade family).
   Fast trend is just the *same* trend edge sampled faster and taxed harder.

3. **A no-trade buffer cuts turnover 3–4× with no net-Sharpe loss** — the key
   lever for any spread-constrained fast book. But it slows the book down, and
   the slower it gets the more it correlates with the champion (0.66→0.80).

4. **A fast sleeve does NOT improve the combined book.** Champion alone
   (vectorized proxy) = 1.28; every 50/50 combo with a fast sleeve is *lower*
   (best 1.23). On one asset a faster book is a correlated (0.73–0.80), weaker
   clone — its lower standalone Sharpe isn't offset by diversification. This
   re-confirms: the jump past ~1.06 is **multi-asset**, not faster single-XAU.

5. **Discrete engine demolishes the vectorized illusion.** The continuous
   vol-target (net SR ~1.0) collapses to **net SR 0.4–0.5** once real 3×ATR
   stops (whipsaw, 37–39% win), lot quantization and $0.34 spread apply.
   Per-year: **negative in 2017/2018/2021**, the whole edge from the 2024–26
   gold bull → it is leveraged bull-beta, not a robust standalone edge.
   (`flip` mode "0.80 / 1 trade" is a degenerate artifact = buy-and-hold, since
   a long-only signal never flips short.)

6. **The killer is the cent spread, and it is account-type-fixable.** M30 fast
   champion, discrete, ~19 trades/mo:

   | Cost regime            | net Sharpe | maxDD  | verdict            |
   |------------------------|-----------:|-------:|--------------------|
   | $0.34 (cent, live now) | **0.50**   | −37.5% | dead               |
   | $0.12 (raw / ECN)      | **0.85**   | −27.2% | deployable standalone |
   | $0.02 (near-zero)      | **1.07**   | −24.5% | confirms gross edge |

   Halving the spread nearly doubles net Sharpe. H1 is less turnover-sensitive
   (0.42→0.58) but lower ceiling; **M30 fast is the best fast config.**

## Verdict

- **On the cent account ($0.34): NOT deployable.** A fast XAU trend runner
  nets ~0.5 Sharpe with −37% DD and loses in choppy years. Do not add it to
  the cent book — the champion is already near the single-XAU ceiling and a
  fast clone drags the combined Sharpe down.
- **The "small trend runner" is real only on a raw/ECN gold account (~$0.12)**,
  where M30 fast champion is a legit ~0.85-Sharpe, ~19-trades/mo standalone
  bot (more active than the champion's 3–4/mo). Even there it does not improve
  a champion+fast portfolio (correlated) — it is an *activity* play, not a
  quality upgrade, and it is gold-bull-concentrated (worst year −0.54).
- **The genuine lever for more trades + diversification is cross-ASSET**, not
  cross-speed: the same trend recipe on a few uncorrelated instruments
  (BTC/NDX are the top adds — see memory `xau-best-diversifiers`). The cent
  account can't trade them; FX majors are dead 2016+.

## LIVE VPS VERIFICATION (2026-07-17)

Measured the broker's REAL gold spread on the VPS (68.183.91.240) via the rpyc
bridges, read-only (`scripts/vps_spread_probe.py`, `vps_symbol_tradability.py`):

| account / group          | symbol   | spread $/oz | tradable | contract |
|--------------------------|----------|------------:|----------|---------:|
| LIVE CENT 54939391 (Live2)| XAUUSDc | **0.36**    | FULL     | 1 oz     |
| DEMO 57482374 (Standard) | XAUUSD   | 0.36        | FULL     | 100 oz   |
| DEMO Standard            | **XAUUSDb** | **0.10** | **FULL** | 100 oz   |
| DEMO Standard            | XAUUSDr  | 0.16        | DISABLED | 100 oz   |
| DEMO Standard            | XAUUSD.F | 0.60        | DISABLED | 10 oz    |

- **Live cent XAUUSDc = $0.36** (rock-steady) — confirms & slightly WORSENS the
  $0.34 assumption. Re-running discrete at $0.36 → net SR **0.49** (DD −36%,
  red 2017/18/21). Fast trend is VERIFIED DEAD on the live cent account.
- **The raw-tier symbols are NOT visible on the live cent group** (XAUUSDr/b
  return no info on 18813) → no tight-spread path on the current live account.
- **XAUUSDb ($0.10) is FULL-tradable on the HFM demo** → the tight gold spread is
  real at HFM, just gated behind a different account type. Discrete at the
  measured $0.10 → net SR **0.89**; at $0.16 (spread + est. commission) → **0.80**.
- Swap on gold = −71.9 pts/oz/night long (−$0.72/oz) — NOT modeled; a fast bot
  flat overnight avoids it, a trailing multi-day hold pays it. Caveat for any
  build.

**Net:** verification upholds every conclusion. Deployable ONLY on a raw-tier
HFM gold account (~$0.10–0.16), which the live cent account cannot reach.

## SIZING / SELECTION + IMPLEMENTATION (2026-07-17)

Does sizing beat the spread? Tested (`scripts/v5_xau_fast_trend_sizing.py`,
$0.36):
- **Uniform/flat sizing is Sharpe-neutral-to-harmful** (flat 0.39 < baseline
  0.49) — cannot lever out of a cost problem (Sharpe is scale-invariant).
- **State-dependent sizing DOES help: 0.49 → 0.75.** Lever = conviction
  concentration + trade SELECTION (skip weak trades that don't clear the
  spread). High-conviction-only (enter_thresh 1.5) is best risk-adjusted.
- But it converges toward the champion (fewer, bigger trades) and never beats
  it on the cent account; worst-year stays ~−1.0 (edge-quality, not sizing).

**Spread frontier** (`v5_xau_fast_trend_spread_frontier.py`, conviction-selected,
best threshold per spread):

| spread $/oz | net Sharpe | verdict |
|---|---:|---|
| 0.10 | 1.07 | strong |
| 0.14 | 1.04 | beats champion |
| 0.18 | 0.99 | ~champion |
| **0.24** | **0.91** | **break-even vs champion** |
| 0.30 | 0.81 | marginal |
| 0.36 (cent) | 0.75 | below champion |

=> **REQUIRED ACCOUNT: raw/ECN gold spread ≤ $0.24/oz (≤ $0.14 to beat the
champion).**

**IMPLEMENTED (magic 360543), NOT ENABLED:**
- `src/v5/xau_fast_signals.py` — fast_champion_signal (M30, byte-identical to
  research; signal parity max|diff|=0.0).
- `scripts/v5_xau_fast.py` — M30 executor forked from v5_xau_dual, with a
  **spread_guard preflight** that ABORTS if live spread > max_spread_usd ($0.24)
  → physically cannot bleed on the cent account.
- `configs/v5_xau_fast.json` (enter 1.5, risk_frac 0.5% → ~−16% DD at $0.12),
  `scripts/xau_fast_cron.sh`, `deploy/xau-fast.{service,timer}` (disabled).
- **VPS dry-run verified (demo 57482374):** default XAUUSD → guard ABORTS
  ($0.36>$0.24); `XAUUSDb` ($0.10) → guard passes, M30 merge + plan OK, magic
  360543 isolated, no orders sent. Deployed-config backtest: $0.12 → SR 1.05 /
  −16.5% DD (risk 0.5%).

**To go live:** open an HFM raw/ECN gold account (≤$0.24), set
`bots.fast.symbol_override` to its raw symbol (e.g. XAUUSDb), point .env at it,
enable AutoTrading, then `systemctl --user enable --now xau-fast.timer`.

Artifacts: `sweep_results.csv`, `fast-trend-discrete-*` run dirs, VPS probes
`scripts/vps_spread_probe.py` + `vps_symbol_tradability.py`. Engines causal
(closes≤t, shift(1) scalars, next-bar fills).
