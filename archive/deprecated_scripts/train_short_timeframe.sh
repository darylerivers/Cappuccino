#!/bin/bash
# Train models on shorter timeframes for more frequent trading
# Usage: ./train_short_timeframe.sh [5m|15m|30m] [workers]

set -e

TIMEFRAME=${1:-"15m"}
WORKERS=${2:-2}

echo "========================================"
echo "Cappuccino Short Timeframe Training"
echo "========================================"
echo ""
echo "Timeframe: $TIMEFRAME"
echo "Workers: $WORKERS"
echo ""

# Validate timeframe
if [[ ! "$TIMEFRAME" =~ ^(5m|15m|30m|1h)$ ]]; then
    echo "Error: Invalid timeframe '$TIMEFRAME'"
    echo "Supported: 5m, 15m, 30m, 1h"
    exit 1
fi

# Create config file for short timeframe
CONFIG_FILE="config_${TIMEFRAME}.py"

cat > "$CONFIG_FILE" << 'PYTHON_EOF'
"""
Configuration for {TIMEFRAME} timeframe training
Auto-generated for more frequent trading
"""

from datetime import datetime, timedelta
import numpy as np
import operator as op
from functools import reduce


def nCr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom


# General Training Settings
trade_start_date = '2023-10-02 00:00:00'
trade_end_date = '2025-10-02 00:00:00'

TRAIN_START_OVERRIDE = None
VAL_START_OVERRIDE = None

SEED_CFG = 2390408
DATA_SOURCE = 'alpaca'
TIMEFRAME = '{TIMEFRAME}'  # SHORTER TIMEFRAME FOR FREQUENT TRADING
H_TRIALS = 150
KCV_groups = 5
K_TEST_GROUPS = 2
NUM_PATHS = 3
N_GROUPS = NUM_PATHS + 1
NUMBER_OF_SPLITS = nCr(N_GROUPS, N_GROUPS - K_TEST_GROUPS)

# Adjust windows for shorter timeframe
# {TIMEFRAME} = more data points needed
TIMEFRAME_MULTIPLIER = {
    '5m': 12,   # 12x more candles than 1h
    '15m': 4,   # 4x more candles than 1h
    '30m': 2,   # 2x more candles than 1h
    '1h': 1,    # baseline
}

MULTIPLIER = TIMEFRAME_MULTIPLIER['{TIMEFRAME}']

# Scale training windows
TRAIN_WINDOW_HOURS = 1440 * MULTIPLIER  # Scale by timeframe
VAL_WINDOW_HOURS = 240 * MULTIPLIER

PAPER_TRADING_HISTORY_HOURS_MIN = 24
PAPER_TRADING_HISTORY_HOURS_MAX = 120
PAPER_TRADING_HISTORY_HOURS_DEFAULT = 120

CANDLE_TO_MINUTES = {
    '1m': 1,
    '5m': 5,
    '10m': 10,
    '30m': 30,
    '1h': 60,
    '2h': 120,
    '4h': 240,
    '12h': 720,
    '1d': 1440,
}


def _candles_from_hours(hours: int, timeframe: str) -> int:
    minutes = CANDLE_TO_MINUTES.get(timeframe)
    if minutes is None:
        raise ValueError(f"Unsupported timeframe '{timeframe}'.")
    return max(int((hours * 60) / minutes), 1)


COINBASE_WEBSOCKET_ENABLED = False
COINBASE_WEBSOCKET_DURATION_SECONDS = 120
COINBASE_WEBSOCKET_INCLUDE_OPEN_BUCKET = True

TRAINVAL_USE_TRADE_RANGE = False

print(NUMBER_OF_SPLITS)

no_candles_for_train = _candles_from_hours(TRAIN_WINDOW_HOURS, TIMEFRAME)
no_candles_for_val = _candles_from_hours(VAL_WINDOW_HOURS, TIMEFRAME)

# Same tickers as main config
TICKER_LIST = ['AAVE/USD',
               'AVAX/USD',
               'BTC/USD',
               'LINK/USD',
               'ETH/USD',
               'LTC/USD',
               'UNI/USD',
               ]

# Minimum buy limits
ALPACA_LIMITS = np.array([0.005,    # AAVE
                          0.03,     # AVAX
                          0.00001,  # BTC
                          0.1,      # LINK
                          0.0003,   # ETH
                          0.01,     # LTC
                          0.2,      # UNI
                          ])

TECHNICAL_INDICATORS_LIST = ['open',
                             'high',
                             'low',
                             'close',
                             'volume',
                             'macd',
                             'macd_signal',
                             'macd_hist',
                             'rsi',
                             'cci',
                             'dx'
                             ]


def calculate_start_end_dates(candlewidth):
    from datetime import datetime, timedelta

    if candlewidth not in CANDLE_TO_MINUTES:
        raise ValueError('Timeframe not supported yet, please manually add!')
    no_minutes = CANDLE_TO_MINUTES[candlewidth]

    trade_start_date_dt = datetime.strptime(trade_start_date, "%Y-%m-%d %H:%M:%S")
    trade_end_date_dt = datetime.strptime(trade_end_date, "%Y-%m-%d %H:%M:%S")

    train_override_dt = datetime.strptime(TRAIN_START_OVERRIDE, "%Y-%m-%d %H:%M:%S") if TRAIN_START_OVERRIDE else None
    val_override_dt = datetime.strptime(VAL_START_OVERRIDE, "%Y-%m-%d %H:%M:%S") if VAL_START_OVERRIDE else None

    if train_override_dt or val_override_dt:
        global no_candles_for_train, no_candles_for_val

        if val_override_dt is None:
            val_override_dt = trade_start_date_dt - timedelta(minutes=no_minutes * no_candles_for_val)

        if train_override_dt is None:
            train_override_dt = val_override_dt - timedelta(minutes=no_minutes * no_candles_for_train)

        train_start_date_dt = train_override_dt
        val_start_date_dt = val_override_dt
        train_end_date_dt = val_start_date_dt - timedelta(minutes=no_minutes)
        val_end_date_dt = trade_end_date_dt - timedelta(minutes=no_minutes)

        def _candles_between(start_dt, end_dt):
            total_minutes = (end_dt - start_dt).total_seconds() / 60
            return max(int(total_minutes // no_minutes), 0)

        no_candles_for_train = max(_candles_between(train_start_date_dt, val_start_date_dt), 1)
        no_candles_for_val = max(_candles_between(val_start_date_dt, trade_end_date_dt), 1)

        return (
            train_start_date_dt.strftime("%Y-%m-%d %H:%M:%S"),
            train_end_date_dt.strftime("%Y-%m-%d %H:%M:%S"),
            val_start_date_dt.strftime("%Y-%m-%d %H:%M:%S"),
            val_end_date_dt.strftime("%Y-%m-%d %H:%M:%S"),
        )

    train_start_date = (trade_start_date_dt
                        - timedelta(minutes=no_minutes * (no_candles_for_train
                                                          + no_candles_for_val))).strftime("%Y-%m-%d %H:%M:%S")

    train_end_date = (trade_start_date_dt
                      - timedelta(minutes=no_minutes * (no_candles_for_val + 1))).strftime("%Y-%m-%d %H:%M:%S")

    val_start_date = (trade_start_date_dt
                      - timedelta(minutes=no_minutes * no_candles_for_val)).strftime("%Y-%m-%d %H:%M:%S")

    val_end_date = (trade_start_date_dt
                    - timedelta(minutes=no_minutes * 1)).strftime("%Y-%m-%d %H:%M:%S")

    return train_start_date, train_end_date, val_start_date, val_end_date


TRAIN_START_DATE, TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE = calculate_start_end_dates(TIMEFRAME)
PYTHON_EOF

# Replace placeholder with actual timeframe
sed -i "s/{TIMEFRAME}/$TIMEFRAME/g" "$CONFIG_FILE"

echo "✓ Created configuration file: $CONFIG_FILE"
echo ""

# Download data for this timeframe
echo "Step 1: Downloading $TIMEFRAME data..."
python3 << PYTHON_DOWNLOAD
import sys
sys.path.insert(0, '.')
from $CONFIG_FILE import *
from 0_dl_trainval_data import download_trainval_data

print(f"Downloading data for {TIMEFRAME} timeframe...")
print(f"Tickers: {TICKER_LIST}")
print(f"Training candles: {no_candles_for_train}")
print(f"Validation candles: {no_candles_for_val}")

download_trainval_data()
print("✓ Data downloaded successfully")
PYTHON_DOWNLOAD

echo ""
echo "Step 2: Starting training with $WORKERS workers..."
echo "Study name: cappuccino_${TIMEFRAME}_$(date +%Y%m%d_%H%M)"
echo ""

# Start training
for i in $(seq 1 $WORKERS); do
    echo "Starting worker $i..."
    nohup python3 1_optimize_unified.py \
        --config "$CONFIG_FILE" \
        --study-name "cappuccino_${TIMEFRAME}_$(date +%Y%m%d_%H%M)" \
        --trials 50 \
        > "logs/training_${TIMEFRAME}_worker${i}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

    WORKER_PID=$!
    echo "  Worker $i started (PID: $WORKER_PID)"
    sleep 2
done

echo ""
echo "========================================"
echo "Training Started!"
echo "========================================"
echo ""
echo "Monitor progress:"
echo "  python3 dashboard_training_detailed.py"
echo ""
echo "View logs:"
echo "  tail -f logs/training_${TIMEFRAME}_worker*.log"
echo ""
echo "Stop training:"
echo "  pkill -f '1_optimize_unified.py.*${TIMEFRAME}'"
echo ""
