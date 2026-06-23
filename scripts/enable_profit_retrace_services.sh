#!/usr/bin/env bash
set -euo pipefail

write_override() {
  local service="$1"
  local exec_start="$2"
  local dir="/etc/systemd/system/${service}.service.d"

  mkdir -p "$dir"
  cat >"${dir}/profit-retrace.conf" <<EOF
[Service]
ExecStart=
ExecStart=${exec_start}
EOF
}

write_override "mt5-eurusd" "/home/rock/anaconda3/envs/envmt5/bin/python src/bots/pipeline_bot.py \\
    --symbol EURUSD \\
    --model-dir data/models/pipeline_EURUSD \\
    --flip-mode profit_retrace \\
    --profit-retrace-activation 120 \\
    --profit-retrace-ratio 0.5 \\
    --tick-interval 5 \\
    --candle-model-dir data/models/candle_EURUSD"

write_override "mt5-eurusd-hedge" "/home/rock/anaconda3/envs/envmt5/bin/python src/bots/pipeline_bot.py \\
    --symbol EURUSD \\
    --model-dir data/models/pipeline_EURUSD_v2 \\
    --candle-feature-dir data/models/candle_EURUSD \\
    --flip-mode profit_retrace \\
    --profit-retrace-activation 120 \\
    --profit-retrace-ratio 0.5 \\
    --tick-interval 5 \\
    --magic 20260102"

write_override "mt5-usdjpy" "/home/rock/anaconda3/envs/envmt5/bin/python src/bots/pipeline_bot.py \\
    --symbol USDJPY \\
    --model-dir data/models/pipeline_USDJPY \\
    --flip-mode profit_retrace \\
    --profit-retrace-activation 120 \\
    --profit-retrace-ratio 0.5 \\
    --tick-interval 5 \\
    --candle-model-dir data/models/candle_USDJPY"

write_override "mt5-usdjpy-hedge" "/home/rock/anaconda3/envs/envmt5/bin/python src/bots/pipeline_bot.py \\
    --symbol USDJPY \\
    --model-dir data/models/pipeline_USDJPY_v2 \\
    --candle-feature-dir data/models/candle_USDJPY \\
    --flip-mode profit_retrace \\
    --profit-retrace-activation 120 \\
    --profit-retrace-ratio 0.5 \\
    --tick-interval 5 \\
    --magic 20260103"

systemctl daemon-reload
systemctl restart mt5-eurusd mt5-eurusd-hedge mt5-usdjpy mt5-usdjpy-hedge
systemctl --no-pager --full status mt5-eurusd mt5-eurusd-hedge mt5-usdjpy mt5-usdjpy-hedge
