"""
This code defines various settings and functions for training a machine learning model.

The trade start date in combinations with the amount of candles required for training and validation determine all
other parameters automatically.

The function nCr calculates the number of ways to choose r elements from a set of n elements, also known as a combination.

The settings defined in this script include the random seed SEED_CFG, the time frame TIMEFRAME,
the number of trials H_TRIALS,
the number of groups used for testing K_TEST_GROUPS,
the number of paths NUM_PATHS,
the number of K-fold cross validation groups KCV_groups
the number of groups N_GROUPS,
the number of splits NUMBER_OF_SPLITS,
the start and end date for the trade period trade_start_date
the trade_end_date,
the number of candles for training no_candles_for_train
the validation no_candles_for_val
the list of tickers TICKER_LIST,
the minimum buy limits ALPACA_LIMITS,
the list of technical indicators TECHNICAL_INDICATORS_LIST.

The function calculate_start_end_dates is used to compute the start and end dates for training and validation based on the number of candles and the selected time frame.

"""

from datetime import datetime, timedelta
import numpy as np
import operator as op
from functools import reduce
from constants import TRAINING, DATA


def nCr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2


# General Training Settings
#######################################################################################################
#######################################################################################################

trade_start_date = '2023-10-02 00:00:00'
trade_end_date = '2025-10-02 00:00:00'

# Optional overrides: set to a timestamp string ("YYYY-MM-DD HH:MM:SS") to
# derive candle counts automatically from explicit calendar boundaries.
# Leave as ``None`` to rely on the rolling-window candle counts below.
TRAIN_START_OVERRIDE = None
VAL_START_OVERRIDE = None

SEED_CFG = TRAINING.SEED  # Import from constants
DATA_SOURCE = 'alpaca'  # Options: 'alpaca', 'binance', 'yahoo' (alpaca for training data)
TIMEFRAME = DATA.TIMEFRAME_DEFAULT  # Import from constants
H_TRIALS = TRAINING.N_TRIALS  # Import from constants
KCV_groups = TRAINING.KCV_GROUPS  # Import from constants
K_TEST_GROUPS = TRAINING.K_TEST_GROUPS  # Import from constants
NUM_PATHS = TRAINING.NUM_PATHS  # Import from constants
N_GROUPS = NUM_PATHS + 1
NUMBER_OF_SPLITS = nCr(N_GROUPS, N_GROUPS - K_TEST_GROUPS)
# print(NUMBER_OF_SPLITS)  # Commented to reduce noise

# Rolling windows tuned for intraday re-training flows
# Increased windows to give agent more data for learning better strategies
TRAIN_WINDOW_HOURS = TRAINING.TRAIN_WINDOW_HOURS  # Import from constants
VAL_WINDOW_HOURS = TRAINING.VAL_WINDOW_HOURS  # Import from constants

# Paper trading history (kept in config_main for backward compatibility)
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

# Coinbase data ingestion preferences
#######################################################################################################
#######################################################################################################
COINBASE_WEBSOCKET_ENABLED = False
COINBASE_WEBSOCKET_DURATION_SECONDS = 120  # seconds of real-time candles to stream per download
COINBASE_WEBSOCKET_INCLUDE_OPEN_BUCKET = True

# Training/validation data download preferences
#######################################################################################################
#######################################################################################################
TRAINVAL_USE_TRADE_RANGE = False

no_candles_for_train = _candles_from_hours(TRAIN_WINDOW_HOURS, TIMEFRAME)
no_candles_for_val = _candles_from_hours(VAL_WINDOW_HOURS, TIMEFRAME)

# Import data constants (canonical source of truth)
TICKER_LIST = list(DATA.DEFAULT_TICKERS)
ALPACA_LIMITS = np.array(DATA.ALPACA_LIMITS)
TECHNICAL_INDICATORS_LIST = list(DATA.TECH_INDICATORS)


# Auto compute all necessary dates based on candle distribution
#######################################################################################################
#######################################################################################################

def calculate_start_end_dates(candlewidth):
    no_minutes = int

    if candlewidth not in CANDLE_TO_MINUTES:
        raise ValueError('Timeframe not supported yet, please manually add!')
    no_minutes = CANDLE_TO_MINUTES[candlewidth]

    trade_start_date_datetimeObj = datetime.strptime(trade_start_date, "%Y-%m-%d %H:%M:%S")
    trade_end_date_datetimeObj = datetime.strptime(trade_end_date, "%Y-%m-%d %H:%M:%S")

    train_override_dt = datetime.strptime(TRAIN_START_OVERRIDE, "%Y-%m-%d %H:%M:%S") if TRAIN_START_OVERRIDE else None
    val_override_dt = datetime.strptime(VAL_START_OVERRIDE, "%Y-%m-%d %H:%M:%S") if VAL_START_OVERRIDE else None

    # If overrides are provided, compute candle counts and derived dates from them.
    if train_override_dt or val_override_dt:
        global no_candles_for_train, no_candles_for_val

        if val_override_dt is None:
            val_override_dt = trade_start_date_datetimeObj - timedelta(minutes=no_minutes * no_candles_for_val)

        if train_override_dt is None:
            train_override_dt = val_override_dt - timedelta(minutes=no_minutes * no_candles_for_train)

        train_start_date_dt = train_override_dt
        val_start_date_dt = val_override_dt
        train_end_date_dt = val_start_date_dt - timedelta(minutes=no_minutes)
        val_end_date_dt = trade_end_date_datetimeObj - timedelta(minutes=no_minutes)

        def _candles_between(start_dt, end_dt):
            total_minutes = (end_dt - start_dt).total_seconds() / 60
            return max(int(total_minutes // no_minutes), 0)

        no_candles_for_train = max(_candles_between(train_start_date_dt, val_start_date_dt), 1)
        no_candles_for_val = max(_candles_between(val_start_date_dt, trade_end_date_datetimeObj), 1)

        return (
            train_start_date_dt.strftime("%Y-%m-%d %H:%M:%S"),
            train_end_date_dt.strftime("%Y-%m-%d %H:%M:%S"),
            val_start_date_dt.strftime("%Y-%m-%d %H:%M:%S"),
            val_end_date_dt.strftime("%Y-%m-%d %H:%M:%S"),
        )

    # train start date = trade_start_date - (no_c_t  + no_c_v)
    train_start_date = (trade_start_date_datetimeObj
                        - timedelta(minutes=no_minutes * (no_candles_for_train
                                                          + no_candles_for_val))).strftime("%Y-%m-%d %H:%M:%S")

    # train start date = trade_start_date - (no_c_v + 1)
    train_end_date = (trade_start_date_datetimeObj
                      - timedelta(minutes=no_minutes * (no_candles_for_val + 1))).strftime("%Y-%m-%d %H:%M:%S")

    # validation start date = trade_start_date - no_c_v
    val_start_date = (trade_start_date_datetimeObj
                      - timedelta(minutes=no_minutes * no_candles_for_val)).strftime("%Y-%m-%d %H:%M:%S")

    # validation start date = trade_start_date - 1
    val_end_date = (trade_start_date_datetimeObj
                    - timedelta(minutes=no_minutes * 1)).strftime("%Y-%m-%d %H:%M:%S")

    return train_start_date, train_end_date, val_start_date, val_end_date


TRAIN_START_DATE, TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE = calculate_start_end_dates(TIMEFRAME)
# Note: These dates are only used for paper trading workflows, not rolling window training
# print("TRAIN_START_DATE: ", TRAIN_START_DATE)
# print("VAL_END_DATE: ", VAL_END_DATE)
