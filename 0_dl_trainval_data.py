"""Download training/validation data and persist the processor outputs.

The processor is picked dynamically using ``config_main.DATA_SOURCE`` so Binance tooling is only imported when needed.
"""

import os
import pickle

from config_main import (
    DATA_SOURCE,
    TRAINVAL_USE_TRADE_RANGE,
    trade_start_date,
    trade_end_date,
    COINBASE_WEBSOCKET_DURATION_SECONDS,
    COINBASE_WEBSOCKET_ENABLED,
    COINBASE_WEBSOCKET_INCLUDE_OPEN_BUCKET,
    TICKER_LIST,
    TIMEFRAME,
    no_candles_for_train,
    no_candles_for_val,
    TECHNICAL_INDICATORS_LIST,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    VAL_START_DATE,
    VAL_END_DATE,
)

def print_config_variables():
    print('\n')
    print('TIMEFRAME:                ', TIMEFRAME)
    print('no_candles_for_train:     ', no_candles_for_train)
    print('no_candles_for_val:       ', no_candles_for_val)
    if TRAINVAL_USE_TRADE_RANGE:
        print('DOWNLOAD START DATE:      ', trade_start_date)
        print('DOWNLOAD END DATE:        ', trade_end_date)
    else:
        print('TRAIN_START_DATE:         ', TRAIN_START_DATE)
        print('TRAIN_END_DATE:           ', TRAIN_END_DATE)
        print('VAL_START_DATE:           ', VAL_START_DATE)
        print('VAL_END_DATE:             ', VAL_END_DATE)
    print('TICKER LIST:              ', TICKER_LIST, '\n')


def process_data():
    source = DATA_SOURCE.lower()
    if source == 'alpaca':
        from processor_Alpaca import AlpacaProcessor

        data_processor = AlpacaProcessor()
        run_kwargs = {}
    elif source == 'coinbase':
        from processor_Coinbase import CoinbaseProcessor

        data_processor = CoinbaseProcessor()
        run_kwargs = {}
        if COINBASE_WEBSOCKET_ENABLED:
            run_kwargs = {
                "prefer_websocket": True,
                "websocket_duration_seconds": COINBASE_WEBSOCKET_DURATION_SECONDS,
                "websocket_include_open_bucket": COINBASE_WEBSOCKET_INCLUDE_OPEN_BUCKET,
            }
    elif source == 'binance':
        from processor_Binance import BinanceProcessor

        data_processor = BinanceProcessor()
        run_kwargs = {}
    elif source == 'yahoo':
        from processor_Yahoo import Yahoofinance

        if TRAINVAL_USE_TRADE_RANGE:
            start = trade_start_date
            end = trade_end_date
        else:
            start = TRAIN_START_DATE
            end = VAL_END_DATE

        data_processor = Yahoofinance(
            data_source='yahoo',
            start_date=start,
            end_date=end,
            time_interval=TIMEFRAME
        )

        # Yahoo processor doesn't have a run() method, use individual methods
        data_processor.download_data(TICKER_LIST)
        data_processor.clean_data()
        data_processor.add_technical_indicator(TECHNICAL_INDICATORS_LIST, select_stockstats_talib=1)

        data_from_processor = data_processor.dataframe
        # Base class df_to_array returns (price_array, tech_array, risk_array) - we use risk_array as time_array
        price_array, tech_array, risk_array = data_processor.df_to_array(TECHNICAL_INDICATORS_LIST, if_vix=False)

        # Extract time array from dataframe
        time_array = data_processor.dataframe[data_processor.dataframe.tic == TICKER_LIST[0]]['time'].values

        return data_from_processor, price_array, tech_array, time_array
    else:
        raise ValueError(f"Unsupported DATA_SOURCE '{DATA_SOURCE}'.")

    if TRAINVAL_USE_TRADE_RANGE:
        start = trade_start_date
        end = trade_end_date
    else:
        start = TRAIN_START_DATE
        end = VAL_END_DATE

    data_from_processor, price_array, tech_array, time_array = data_processor.run(
        TICKER_LIST,
        start,
        end,
        TIMEFRAME,
        TECHNICAL_INDICATORS_LIST,
        if_vix=False,
        **run_kwargs,
    )
    return data_from_processor, price_array, tech_array, time_array


def save_data(data_folder, data_from_processor, price_array, tech_array, time_array):
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    with open(data_folder + '/data_from_processor', 'wb') as handle:
        pickle.dump(data_from_processor, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(data_folder + '/price_array', 'wb') as handle:
        pickle.dump(price_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(data_folder + '/tech_array', 'wb') as handle:
        pickle.dump(tech_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(data_folder + '/time_array', 'wb') as handle:
        pickle.dump(time_array, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_data_to_disk(data_from_processor, price_array, tech_array, time_array):
    data_folder = f'./data/{TIMEFRAME}_{no_candles_for_train + no_candles_for_val}'
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    _save_to_disk(data_from_processor, f"{data_folder}/data_from_processor")
    _save_to_disk(price_array, f"{data_folder}/price_array")
    _save_to_disk(tech_array, f"{data_folder}/tech_array")
    _save_to_disk(time_array, f"{data_folder}/time_array")


def _save_to_disk(data, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    print_config_variables()
    data_from_processor, price_array, tech_array, time_array = process_data()
    save_data_to_disk(data_from_processor, price_array, tech_array, time_array)


if __name__ == "__main__":
    main()
