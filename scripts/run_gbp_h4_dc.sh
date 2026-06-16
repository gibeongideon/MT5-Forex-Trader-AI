#!/bin/bash
set -u; PY=/home/rock/anaconda3/envs/envmt5/bin/python
cd /home/rock/Desktop/2026_Projects/MT5; LOG=data/retest_logs
for SEG in discover confirm; do
  echo "=== GBPUSD H4 xgb $SEG $(date) ==="
  $PY -u scripts/backtest_meta_labeling.py --symbol GBPUSD --primary xgb --sweep \
      --data "data/GBPUSD_H4_${SEG}.csv" > "$LOG/gbp_h4_${SEG}.log" 2>&1
  echo "=== DONE $SEG rc=$? $(date) ==="
done
echo "GBP_H4_DC_DONE $(date)"
