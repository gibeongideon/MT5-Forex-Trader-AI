#!/bin/bash
set -u
cd /home/trader/MT5 || exit 1
export MT5_BRIDGE_PORT=18814
/home/trader/miniconda3/envs/envmt5/bin/python scripts/v5_basket_challenge_exec.py \
  --config configs/v5_ftmo_challenge.json \
  --state  data/v5_runs/ftmo_state.json \
  --paper-csv data/v5_runs/ftmo_guard_log.csv \
  --guard-only --live --execute >> data/v5_runs/ftmo-guard.log 2>&1
