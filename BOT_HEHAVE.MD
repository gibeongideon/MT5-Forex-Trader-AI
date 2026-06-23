Current behavior now:
Only the hedge bots are running:
mt5-eurusd-hedge
mt5-usdjpy-hedge

If an old trade is losing, for example:
old USDJPY BUY is -50
new signal opens/keeps a newer USDJPY SELL
Then both can stay open.

The old losing trade is marked as a recovery trade.
It is checked every 5 seconds.
As soon as it reaches >= 0, it closes.
It does not need to reach +120.

The +120 then close at 50% retrace rule now applies only to the latest non-recovery trade.
Example: latest SELL reaches +120, arms tracker.
If it later peaks at +800, exit becomes +400.
Old recovery trades do not use this 120/60 rule.

Same-direction duplicate prevention remains:
If a SELL already exists, the bot skips opening another SELL.
If a BUY exists and the new signal is SELL, that can open/continue as the hedge case.