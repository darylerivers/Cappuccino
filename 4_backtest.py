"""
This script contains a set of functions for loading and processing data for trading.

It contains the following functions:

load_validated_model: Loads the best trial from the pickle file in the specified directory and returns the best trial's attributes.
download_CVIX: Downloads the CVIX dataframe from Yahoo finance and returns it.
load_and_process_data: loads and process the trade data from the specified data folder and returns the data.

After that, the large loop analyzes every result by creating an instance of an Alpaca environment and checking
what the model would do through the environment using the new trading data

Finally, the resulting backtests are analyzes for performance a performance metric per benchmark (EQW, S&P BCI) plus
all the input DRL agents are analyzed.

"""


import argparse
import os
import pickle
from pathlib import Path
import joblib

import numpy as np
import pandas as pd

import matplotlib.dates as mdates

from config_main import *
from function_finance_metrics import *
try:
    from processor_Yahoo import Yahoofinance
except ImportError:
    Yahoofinance = None
from environment_Alpaca import CryptoEnvAlpaca
from drl_agents.elegantrl_models import DRLAgent as DRLAgent_erl


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest trained DRL agents on trade data.")
    parser.add_argument(
        "--results",
        "-r",
        nargs="+",
        help="Result folders (relative to train_results or absolute paths) to backtest. If not provided, auto-detects latest study.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU id to use (-1 for CPU)")
    return parser.parse_args()


def find_latest_study():
    """Auto-detect the most recent optimization study with a best_trial."""
    train_results = Path("./train_results")
    if not train_results.exists():
        return None

    # Find all directories with best_trial files
    candidates = []
    for result_dir in train_results.iterdir():
        if not result_dir.is_dir():
            continue
        best_trial_path = result_dir / "best_trial"
        if best_trial_path.exists():
            candidates.append((result_dir.name, best_trial_path.stat().st_mtime))

    if not candidates:
        return None

    # Sort by modification time (most recent first)
    candidates.sort(key=lambda x: x[1], reverse=True)
    latest_folder = candidates[0][0]
    print(f"Auto-detected latest study: {latest_folder}")
    return latest_folder


def load_validated_model(result):
    result_path = Path(result)
    if not result_path.is_absolute():
        result_path = Path("./train_results") / result_path

    best_trial_path = result_path / "best_trial"
    if not best_trial_path.exists():
        raise FileNotFoundError(f"Could not find best_trial in {result_path}.")

    with open(best_trial_path, 'rb') as handle:
        best_trial = pickle.load(handle)

    print('BEST TRIAL: ', best_trial.number)
    timeframe = best_trial.user_attrs['timeframe']
    ticker_list = best_trial.user_attrs['ticker_list']
    technical_ind = best_trial.user_attrs['technical_indicator_list']
    net_dim = best_trial.params['net_dimension']
    model_name = best_trial.user_attrs['model_name']

    print('\nMODEL_NAME: ', model_name)
    print(best_trial.params)
    print(timeframe)

    name_test = best_trial.user_attrs['name_test']

    env_params = {
        "lookback": best_trial.params['lookback'],
        "norm_cash": 2 ** best_trial.params.get('norm_cash_exp', best_trial.params.get('norm_cash', -11)),
        "norm_stocks": 2 ** best_trial.params.get('norm_stocks_exp', best_trial.params.get('norm_stocks', -8)),
        "norm_tech": 2 ** best_trial.params.get('norm_tech_exp', best_trial.params.get('norm_tech', -14)),
        "norm_reward": 2 ** best_trial.params.get('norm_reward_exp', best_trial.params.get('norm_reward', -9)),
        "norm_action": best_trial.params['norm_action'],
        "time_decay_floor": best_trial.params.get('time_decay_floor', 0.0)
    }

    # Check if this is an intraday retrain (has custom train/test split)
    intraday_retrain = best_trial.user_attrs.get('intraday_retrain', False)
    test_start_idx = best_trial.user_attrs.get('test_start_idx', None)

    return env_params, net_dim, timeframe, ticker_list, technical_ind, name_test, model_name, result_path, intraday_retrain, test_start_idx


def download_CVIX(trade_start_date, trade_end_date):
    if Yahoofinance is None:
        print('Warning: processor_Yahoo not available; skipping CVIX download.')
        return None

    trade_start_date_fmt = trade_start_date[:10]
    trade_end_date_fmt = trade_end_date[:10]
    cache_dir = Path('data/.yfinance_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('YFINANCE_CACHE_DIR', str(cache_dir.resolve()))
    os.environ.setdefault('YFINANCE_CACHE_DISABLE', '1')

    intervals = ['60m', '1h', '1d']
    for interval in intervals:
        YahooProcessor = Yahoofinance('yahoofinance', trade_start_date_fmt, trade_end_date_fmt, interval)
        try:
            CVOL_df = YahooProcessor.download_data(['CVOL-USD'])
        except Exception as err:
            print(f"Warning: failed to download CVIX at interval {interval}: {err}")
            continue

        if CVOL_df.empty:
            print(f"Warning: CVIX data empty at interval {interval}")
            continue

        CVOL_df.set_index('date', inplace=True)
        CVOL_df.index = pd.to_datetime(CVOL_df.index)

        series = CVOL_df['close']
        series.name = 'cvix'

        if interval not in ('60m', '1h'):
            # Upsample lower frequency data to 5-minute grid for smoother alignment
            series = series.resample('5Min').ffill().bfill()

        return series

    print('Warning: Yahoo Finance returned no CVIX data; defaulting to zeros.')
    return None


def load_and_process_data(TIMEFRAME, trade_start_date, trade_end_date):
    data_folder = f'./data/trade_data/{TIMEFRAME}_{str(trade_start_date[2:10])}_{str(trade_end_date[2:10])}'
    print(f'\nLOADING DATA FOLDER: {data_folder}\n')

    # Load arrays from .npy files (more compatible)
    price_array = np.load(data_folder + '/price_array.npy')
    tech_array = np.load(data_folder + '/tech_array.npy')
    time_array = np.load(data_folder + '/time_array.npy', allow_pickle=True)

    # For data_from_processor, try feather, then CSV, skip parquet (needs pyarrow)
    try:
        data_from_processor = pd.read_feather(data_folder + '/data_from_processor.feather')
        print('Loaded from .npy + feather files')
    except:
        try:
            data_from_processor = pd.read_csv(data_folder + '/data_from_processor.csv', index_col=0, parse_dates=True)
            print('Loaded from .npy + CSV files')
        except:
            # Last resort: pickle (may fail on version mismatch)
            with open(data_folder + '/data_from_processor', 'rb') as handle:
                data_from_processor = pickle.load(handle)
            print('Loaded from pickle files (compatibility risk)')

    cvix_series = download_CVIX(trade_start_date, trade_end_date)
    time_index = pd.DatetimeIndex(time_array)
    if cvix_series is None:
        cvix_series = pd.Series(0.0, index=time_index, name='cvix')
    else:
        cvix_series = cvix_series.reindex(time_index, method='nearest')
        cvix_series = cvix_series.ffill().bfill()
    cvix_array = cvix_series.values
    if len(cvix_array) == 0:
        raise ValueError("Merged CVIX series is empty; check trade date window and Yahoo data availability.")
    cvix_array_growth = np.diff(cvix_array, prepend=cvix_array[0])

    return data_from_processor, price_array, tech_array, time_array, cvix_array, cvix_array_growth


# Inputs
#######################################################################################################
#######################################################################################################
#######################################################################################################

print('TRADE_START_DATE             ', trade_start_date)
print('TRADE_END_DATE               ', trade_end_date, '\n')

args = parse_args()
if not args.results:
    latest_study = find_latest_study()
    if latest_study is None:
        raise SystemExit(
            "No trained results found. Please run training first or provide --results argument."
        )
    pickle_results = [latest_study]
    print(f"Using auto-detected study: {latest_study}\n")
else:
    pickle_results = args.results

gpu_id = args.gpu

# Execution
#######################################################################################################
#######################################################################################################

drl_cumrets_list = []
model_names_list = []

_, _, timeframe, ticker_list, technical_ind, _, _, _, intraday_retrain, test_start_idx = load_validated_model(pickle_results[0])
data_from_processor, price_array, tech_array, time_array, cvix_array, cvix_array_growth = load_and_process_data(timeframe, trade_start_date, trade_end_date)

# If this is an intraday retrain, slice data to only use test period
if intraday_retrain and test_start_idx is not None:
    print(f"\n{'='*70}")
    print("INTRADAY BACKTEST MODE: Using only held-out test data")
    print(f"{'='*70}")
    print(f"Original data: {len(price_array)} candles")
    print(f"Test period starts at index: {test_start_idx}")

    price_array = price_array[test_start_idx:]
    tech_array = tech_array[test_start_idx:]
    time_array = time_array[test_start_idx:]
    cvix_array = cvix_array[test_start_idx:]
    cvix_array_growth = cvix_array_growth[test_start_idx:]

    print(f"Test data: {len(price_array)} candles")
    print(f"Test period: {time_array[0]} to {time_array[-1]}")
    print(f"{'='*70}\n")

for count, result in enumerate(pickle_results):
    env_params, net_dim, timeframe, ticker_list, technical_ind, name_test, model_name, result_path, _, _ = load_validated_model(result)
    model_names_list.append(model_name)
    cwd = str((result_path / 'stored_agent').resolve()) + '/'

    data_config = {
        "cvix_array": cvix_array,
        "cvix_array_growth": cvix_array_growth,
        "time_array": time_array,
        "price_array": price_array,
        "tech_array": tech_array,
        "if_train": False,
    }

    env = CryptoEnvAlpaca
    env_instance = env(config=data_config,
                       env_params=env_params,
                       if_log=True
                       )

    account_value_erl = DRLAgent_erl.DRL_prediction(
        model_name=model_name,
        cwd=cwd,
        net_dimension=net_dim,
        environment=env_instance,
        gpu_id=gpu_id
    )

    # Correct slicing (due to DRL start/end)
    lookback = env_params['lookback']
    indice_start = lookback - 1
    indice_end = len(price_array) - lookback
    time_array = time_array[indice_start:indice_end]

    # Slice cvix array
    if count == 0:
        cvix_array = cvix_array[indice_start:indice_end]
        cvix_array_growth = cvix_array_growth[indice_start:indice_end]

    # Compute Sharpe's of each coin
    account_value_eqw, ewq_rets, eqw_cumrets = compute_eqw(price_array, indice_start, indice_end)

    # Compute annualization factor
    data_points_per_year = compute_data_points_per_year(timeframe)
    dataset_size = np.shape(ewq_rets)[0]
    factor = data_points_per_year / dataset_size

    # Compute DRL rets
    account_value_erl = np.array(account_value_erl)
    drl_rets = account_value_erl[1:] - account_value_erl[:-1]
    drl_cumrets = [x / account_value_erl[0] - 1 for x in account_value_erl]
    drl_cumrets_list.append(drl_cumrets)

    # Compute metrics per pickle result
    #######################################################################################################

    # Only compute consistent metrics once
    if count == 0:
        # Load S&P index (if available)
        spy_index_path = Path('data/SPY_Crypto_Broad_Digital_Market_Index - Sheet1.csv')
        if spy_index_path.exists():
            spy_index_df = pd.read_csv(spy_index_path)
            spy_index_df['Date'] = pd.to_datetime(spy_index_df['Date'])

            account_value_spy = np.array(spy_index_df['S&P index'])
            spy_rets = account_value_spy[1:] / account_value_spy[:-1] - 1
            spy_rets = np.insert(spy_rets, 0, 0)
            spy_index_df['cumrets_sp_idx'] = [x / spy_index_df['S&P index'][0] - 1 for x in spy_index_df['S&P index']]
            spy_index_df['rets_sp_idx'] = spy_rets
            spy_index_df.set_index('Date', inplace=True)

            # Use timeframe-aware resampling
            resample_freq = timeframe.replace('m', 'Min').replace('h', 'H').replace('d', 'D')
            spy_index_df = spy_index_df.resample(resample_freq).interpolate(method='pchip')
        else:
            print(f"Warning: S&P index file not found at {spy_index_path}. Skipping S&P benchmark.")
            spy_index_df = None

        if spy_index_df is not None:
            sp_annual_ret, sp_annual_vol, sp_sharpe_rat, sp_vol = aggregate_performance_array(spy_rets, factor)

            write_metrics_to_results('S&P Broad Crypto index',
                                     'plots_and_metrics/test_metrics.txt',
                                     spy_index_df['cumrets_sp_idx'],
                                     sp_annual_ret,
                                     sp_annual_vol,
                                     sp_sharpe_rat,
                                     sp_vol,
                                     'w'
                                     )
            write_mode = 'a'
        else:
            write_mode = 'w'

        # Write buy-and-hold strategy
        eqw_annual_ret, eqw_annual_vol, eqw_sharpe_rat, eqw_vol = aggregate_performance_array(np.array(ewq_rets),
                                                                                                 factor)
        write_metrics_to_results('Buy-and-Hold',
                                 'plots_and_metrics/test_metrics.txt',
                                 eqw_cumrets,
                                 eqw_annual_ret,
                                 eqw_annual_vol,
                                 eqw_sharpe_rat,
                                 eqw_vol,
                                 write_mode
                                 )

    # Then compute the actual metrics from the DRL agents
    drl_annual_ret, drl_annual_vol, drl_sharpe_rat, drl_vol = aggregate_performance_array(np.array(drl_rets), factor)
    write_metrics_to_results(model_name,
                             'plots_and_metrics/test_metrics.txt',
                             drl_cumrets,
                             drl_annual_ret,
                             drl_annual_vol,
                             drl_sharpe_rat,
                             drl_vol,
                             'a'
                             )

    # Hold out of loop only add once
    #######################################################################################################

# Plot
#######################################################################################################
#######################################################################################################

drl_rets_array = np.transpose(np.vstack(drl_cumrets_list))

# General 1
plt.rcParams.update({'font.size': 22})
plt.figure(dpi=300)
f, ax1 = plt.subplots(figsize=(20, 8))

# Plot returns
line_width = 2
if spy_index_df is not None:
    ax1.plot(spy_index_df.index, spy_index_df['cumrets_sp_idx'].values, linewidth=3, label='S&P BDM Index')
ax1.plot(time_array, eqw_cumrets[1:], linewidth=line_width, label='Equal-weight', color='blue')


for i in range(np.shape(drl_rets_array)[1]):
    ax1.plot(time_array, drl_rets_array[:, i], label=model_names_list[i], linewidth=line_width)

n_legends = len(model_names_list) + 1 + (1 if spy_index_df is not None else 0)
ax1.legend(frameon=False, ncol=n_legends, loc='upper left', bbox_to_anchor=(0, 1.11))
ax1.patch.set_edgecolor('black')
ax1.patch.set_linewidth(3)
ax1.grid()

# Plot CVIX
ax2 = ax1.twinx()
ax2.plot(time_array, cvix_array, linewidth=4, label='CVIX', color='black', linestyle='dashed', alpha=0.4)
ax2.legend(frameon=False, loc='upper right', bbox_to_anchor=(0.7, 1.17))
ax2.patch.set_edgecolor('black')
ax2.patch.set_linewidth(3)
ax2.set_ylabel('CVIX')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=8))
ax1.set_ylabel('Cumulative return')
plt.xlabel('Date')
plt.savefig('./plots_and_metrics/test_cumulative_return.png', bbox_inches='tight')
