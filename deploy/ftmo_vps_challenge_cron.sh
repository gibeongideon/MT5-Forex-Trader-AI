#!/bin/bash
# FTMO 2-Step — hourly RECONCILE (XAU+BTC+NDX, magic 360561). Instance 3:
# prefix ~/.mt5c, display :101, bridge 18814. Readiness gate before trading.
set -u
cd /home/trader/MT5 || exit 1
export MT5_BRIDGE_PORT=18814
PY=/home/trader/miniconda3/envs/envmt5/bin/python
LOG=data/v5_runs/ftmo-cron.log
echo "=== $(date -u '+%F %T UTC') ftmo reconcile ===" >> "$LOG"
ready=0
for i in $(seq 1 24); do
  if $PY -c "
import sys
from mt5linux import MetaTrader5
m = MetaTrader5(host='localhost', port=18814)
if not m.initialize() or m.account_info() is None: sys.exit(1)
for s in ('XAUUSD','BTCUSD','US100.cash'):
    m.symbol_select(s, True)
    i=m.symbol_info(s); t=m.symbol_info_tick(s)
    if i is None or t is None or not getattr(t,'bid',0): sys.exit(1)
m.shutdown()
" >/dev/null 2>&1; then ready=1; break; fi
  sleep 5
done
if [ "$ready" -ne 1 ]; then
  echo "$(date -u '+%F %T UTC') NOT READY — skipping" >> "$LOG"; exit 1
fi
$PY scripts/v5_basket_challenge_exec.py \
  --config configs/v5_ftmo_challenge.json \
  --state  data/v5_runs/ftmo_state.json \
  --paper-csv data/v5_runs/ftmo_live_log.csv \
  --live --execute >> "$LOG" 2>&1
echo "$(date -u '+%F %T UTC') ftmo reconcile done" >> "$LOG"
