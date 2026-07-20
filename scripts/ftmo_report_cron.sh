#!/bin/bash
set -u
cd /home/rock/Desktop/2026_Projects/Trader36/MT5 || exit 1
export DISPLAY=:0
CONDA=/home/rock/anaconda3/bin/conda
$CONDA run -n envmt5 python scripts/ftmo_daily_report.py >> data/v5_runs/ftmo-report-cron.log 2>&1
