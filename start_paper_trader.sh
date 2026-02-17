#!/bin/bash
cd /opt/user-data/experiment/cappuccino

~/.pyenv/versions/cappuccino-rocm/bin/python -u scripts/deployment/paper_trader_alpaca_polling.py \
    --model-dir deployments/model_0 \
    --timeframe 1h \
    --history-hours 120 \
    --poll-interval 3600 \
    --gpu 0 \
    > logs/paper_trader.log 2>&1 &

PID=$!
echo "Paper trader started: PID $PID"
echo $PID > logs/paper_trader.pid

sleep 5
echo ""
echo "=== Startup Log ==="
tail -30 logs/paper_trader.log
