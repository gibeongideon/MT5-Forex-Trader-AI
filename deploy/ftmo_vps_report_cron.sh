#!/bin/bash
set -u
cd /home/trader/MT5 || exit 1
export MT5_BRIDGE_PORT=18814
/home/trader/miniconda3/envs/envmt5/bin/python scripts/challenge_daily_report.py \
  --config configs/v5_ftmo_challenge.json \
  --state  data/v5_runs/ftmo_state.json \
  --port 18814 >> data/v5_runs/ftmo-report.log 2>&1
