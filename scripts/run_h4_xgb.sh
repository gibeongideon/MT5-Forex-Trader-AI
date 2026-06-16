#!/bin/bash
set -u
PY=/home/rock/anaconda3/envs/envmt5/bin/python
cd /home/rock/Desktop/2026_Projects/MT5
LOG=data/retest_logs; mkdir -p "$LOG"
echo "=== H4 XGB-PRIMARY FOLLOW-UP START $(date) ==="
for SYM in GBPUSD USDJPY EURUSD; do
  echo "=== RUN $SYM H4 xgb $(date) ==="
  $PY -u scripts/backtest_meta_labeling.py --symbol "$SYM" --primary xgb --sweep \
      --data "data/${SYM}_H4_long.csv" > "$LOG/retest_H4xgb_${SYM}.log" 2>&1
  echo "=== DONE $SYM H4 xgb rc=$? $(date) ==="
done
echo "H4_XGB_DONE $(date)"
