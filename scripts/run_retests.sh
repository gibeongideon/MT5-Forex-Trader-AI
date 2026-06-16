#!/bin/bash
# Autonomous timeframe re-test chain (persistent — survives /tmp clears on reboot).
# Resamples deep M15 _long files to H1/H4, then runs leak-free meta-labeling
# (rule primary, all-hours, sweep, real spread) per timeframe. M15 11yr runs last.
# Usage: bash scripts/run_retests.sh "EURUSD USDJPY GBPUSD"   (default = those 3)
set -u
PY=/home/rock/anaconda3/envs/envmt5/bin/python
cd /home/rock/Desktop/2026_Projects/MT5
SYMS="${1:-EURUSD USDJPY GBPUSD}"
LOGDIR=/home/rock/Desktop/2026_Projects/MT5/data/retest_logs
mkdir -p "$LOGDIR"
echo "=== RETEST CHAIN START $(date) — symbols: $SYMS ==="

# 1. resample to H1 + H4
for SYM in $SYMS; do
  $PY -u scripts/resample_tf.py --symbol "$SYM" --tf H1
  $PY -u scripts/resample_tf.py --symbol "$SYM" --tf H4
done

# 2. H1 then H4 (the new info), then M15-11yr (heavy, last)
for TF in H1 H4 M15; do
  for SYM in $SYMS; do
    f="data/${SYM}_${TF}_long.csv"
    [ -f "$f" ] || { echo "skip $SYM $TF (no file)"; continue; }
    echo "=== RUN $SYM $TF $(date) ==="
    $PY -u scripts/backtest_meta_labeling.py --symbol "$SYM" --primary rule --sweep \
        --data "$f" > "$LOGDIR/retest_${TF}_${SYM}.log" 2>&1
    echo "=== DONE $SYM $TF rc=$? $(date) ==="
  done
done
echo "ALL_RETESTS_DONE $(date)"
