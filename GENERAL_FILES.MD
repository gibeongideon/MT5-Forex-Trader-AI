# General Concepts & How Things Work

> Plain-English explanations of what each part of this project does.
> Written for someone with no trading background.
> No code — just concepts.

---

## What is a Backtest?

Think of it like this: **you rewind history and pretend you were trading.**

You take old price data (e.g. EURUSD prices every 15 minutes from 2024 to 2026) and
run your strategy through it bar by bar — as if you were sitting there in 2024 making
decisions, but using data you already have. You then measure how much money you would
have made or lost.

It answers the question: **"If I had used this strategy in the past, what would have happened?"**

---

## How Our Backtest Works (Step by Step)

### 1. The Data

We have **50,000 price bars** for EURUSD (Euro vs US Dollar).
Each bar = 15 minutes of trading activity. Total coverage: May 2024 → May 2026.

Each bar contains four numbers:
| Name | Meaning |
|------|---------|
| **Open** | Price at the start of that 15-minute window |
| **High** | Highest price reached during those 15 minutes |
| **Low** | Lowest price reached during those 15 minutes |
| **Close** | Price at the end of those 15 minutes |

### 2. Bar-by-Bar Simulation

The backtest walks through each bar **in strict time order**, oldest to newest.
At each bar it asks: *"Does my strategy want to buy, sell, or do nothing right now?"*

It cannot look ahead. It only knows what happened up to that moment — just like real trading.

### 3. Opening a Trade

When the strategy says **BUY** at a given bar, the simulation records:
- **Entry price** — the price at which the trade opens
- **Stop Loss (SL)** — a price level *below* entry; if price falls here, the trade closes automatically to limit the loss
- **Take Profit (TP)** — a price level *above* entry; if price rises here, the trade closes and locks in a profit

For a **SELL** trade, SL is above entry and TP is below — because a sell profits when price falls.

### 4. Checking SL and TP on Every Bar

Once a trade is open, every subsequent bar is checked:
- Did the **Low** of this bar touch or cross the Stop Loss? → **Exit at a loss**
- Did the **High** of this bar touch or cross the Take Profit? → **Exit at a profit**
- Neither? → **Hold the trade and move to the next bar**

### 5. What a "Pip" Is

A pip is the smallest standard price movement in Forex.
For EURUSD, 1 pip = 0.0001 in price terms (the 4th decimal place).

Example: price moves from 1.0850 to 1.0880 = a move of 30 pips.

In our backtest: **SL = 30 pips** (maximum loss per trade), **TP = 60 pips** (target profit).
This is a 2:1 reward-to-risk ratio — you risk 1 to potentially gain 2.

### 6. Position Sizing (How Much to Risk)

We don't bet the same fixed dollar amount every trade. Instead:

> **Risk exactly 1% of current account balance per trade.**

So if your balance is $10,000, you risk $100 on the next trade.
If you win and balance grows to $10,500, you risk $105 on the next trade.
If you lose and balance drops to $9,900, you risk $99 on the next.

This is called **compounding** — losses shrink your bet size, wins grow it. It's realistic.

### 7. Final Report

At the end of all bars, the backtest produces a report:

```
Total trades  : 354
Win rate      : 33.3%  (118 wins / 236 losses)
Profit factor : 0.99
Max drawdown  : 21.6%
Sharpe ratio  : -0.17
Total return  : -2.5%
Final balance : $9,751.94  (started with $10,000)
```

---

## What the Results Mean

| Metric | Plain English |
|--------|--------------|
| **Total trades** | How many times the strategy opened and closed a position |
| **Win rate** | What percentage of trades hit the take profit (vs stop loss) |
| **Profit factor** | Gross profit ÷ gross loss. Above 1.0 = profitable. Below 1.0 = losing |
| **Max drawdown** | The worst peak-to-trough drop in your balance during the test (as %) |
| **Sharpe ratio** | How much return you earn per unit of risk. Higher is better. Below 0 = losing strategy |
| **Total return** | Your overall gain or loss as a percentage over the full test period |

### Our Phase 1 Result Explained

The **random baseline** (trades opened at random with no strategy) produced:
- 33% win rate — roughly 1 in 3 trades profited
- -2.5% total return over 2 years
- Sharpe ratio of -0.17

This is the **expected result for a strategy with no real edge**. The 33% win rate with a 2:1
reward-to-risk ratio should theoretically break even — but the broker's spread (a small cost
per trade) tips it slightly negative. This is intentional: it confirms the baseline has no
predictive power, so anything better in later phases is genuine signal.

---

## Does the Backtest Show "Real" Results?

**Mostly yes.** Here is what it gets right and what it simplifies:

### What is accurate
- Real historical prices, tick by tick (bar by bar)
- SL and TP hit detection uses actual High/Low of each bar — not just closing price
- Balance compounds correctly after each trade
- Position size adjusts to current balance (realistic)

### What is currently simplified (slightly optimistic)

| Missing cost | What it means in real life |
|---|---|
| **Spread** | Brokers charge 1–2 pips per trade just to open it. On 354 trades this is significant. |
| **Slippage** | In fast markets, your order may fill at a slightly worse price than expected. |
| **Commission** | Some brokers charge a fixed fee per lot traded, on top of spread. |

These costs are small individually but matter across hundreds of trades.
**Phase 7** of the implementation plan adds all three costs to the backtest engine,
making the simulation more realistic (and results slightly worse than what Phase 1–2 show).

---

## Why Past Results Don't Guarantee Future Results

The backtest uses **one specific 2-year window** of market history.
Markets constantly change — trending periods, ranging periods, high volatility, low volatility.
A strategy that worked well in 2024–2026 might behave differently in 2027.

This is why the plan includes:
- **Walk-forward validation** (Phase 7) — train on one period, test on a future period you haven't touched
- **Monte Carlo simulation** (Phase 7) — shuffle the order of trades 1,000 times to check if the result was luck
- **Regime detection** — identify whether the market is trending or ranging, and adapt accordingly

The goal is not just "did it work in the past?" but "is it likely to keep working?"

---

## Summary in One Paragraph

> The backtest replays 2 years of real historical price data bar by bar, simulating
> your strategy making buy and sell decisions at the prices that actually existed.
> It tracks every trade opened and closed, computes the profit or loss on each,
> compounds your account balance, and at the end tells you exactly what would have
> happened to your money — including the worst losing streak you would have lived through.
> It is as close to "real trading in the past" as possible, with the small caveat that
> broker costs (spread, slippage, commission) are not yet included and will be added
> in a later phase.
