"""Download trade data in chunks to avoid API rate limits.

This script downloads historical data in smaller monthly chunks and combines them,
avoiding Coinbase API timeout/rate limit issues.

Usage:
    python 0_dl_trade_data_chunked.py
"""

import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

from config_main import (
    DATA_SOURCE,
    TICKER_LIST,
    TECHNICAL_INDICATORS_LIST,
    TIMEFRAME,
    COINBASE_WEBSOCKET_DURATION_SECONDS,
    COINBASE_WEBSOCKET_ENABLED,
    COINBASE_WEBSOCKET_INCLUDE_OPEN_BUCKET,
    trade_start_date,
    trade_end_date,
)


def download_chunk(processor, tickers, start_date, end_date, timeframe, tech_indicators):
    """Download a single chunk of data."""
    print(f'  Downloading {start_date} to {end_date}...')

    try:
        data_from_processor, price_array, tech_array, time_array = processor.run(
            tickers,
            start_date,
            end_date,
            timeframe,
            tech_indicators,
            if_vix=False,
        )
        return data_from_processor, price_array, tech_array, time_array
    except Exception as e:
        print(f'  ERROR downloading chunk: {e}')
        return None, None, None, None


def combine_chunks(chunks):
    """Combine multiple data chunks into single arrays."""
    if not chunks:
        raise ValueError("No data chunks to combine")

    # Filter out failed chunks
    valid_chunks = [(df, pa, ta, tm) for df, pa, ta, tm in chunks if df is not None]

    if not valid_chunks:
        raise ValueError("All chunks failed to download")

    print(f'\nCombining {len(valid_chunks)} chunks...')

    # Combine arrays first (more reliable)
    all_price = [chunk[1] for chunk in valid_chunks]
    all_tech = [chunk[2] for chunk in valid_chunks]
    all_time = [chunk[3] for chunk in valid_chunks]

    combined_price = np.vstack(all_price)
    combined_tech = np.vstack(all_tech)
    combined_time = np.concatenate(all_time)

    # Remove duplicate timestamps
    unique_times, unique_indices = np.unique(combined_time, return_index=True)
    combined_price = combined_price[unique_indices]
    combined_tech = combined_tech[unique_indices]
    combined_time = unique_times

    # Combine dataframes (this is just metadata, arrays are what matter)
    all_dfs = [chunk[0] for chunk in valid_chunks]
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Sort by tic if that column exists
    if 'tic' in combined_df.columns:
        combined_df = combined_df.sort_values('tic').reset_index(drop=True)

    # Remove exact duplicates across all columns
    combined_df = combined_df.drop_duplicates(keep='first').reset_index(drop=True)

    print(f'Combined data: {len(combined_time)} candles')
    print(f'Date range: {combined_time[0]} to {combined_time[-1]}')

    return combined_df, combined_price, combined_tech, combined_time


def generate_date_ranges(start_date, end_date, chunk_months=3):
    """Generate date ranges for chunked downloads."""
    start = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")

    ranges = []
    current = start

    while current < end:
        # Calculate chunk end (either chunk_months ahead or final end_date)
        chunk_end = current + timedelta(days=chunk_months * 30)
        if chunk_end > end:
            chunk_end = end

        ranges.append((
            current.strftime("%Y-%m-%d %H:%M:%S"),
            chunk_end.strftime("%Y-%m-%d %H:%M:%S")
        ))

        current = chunk_end

    return ranges


def save_data_to_disk(data_from_processor, price_array, tech_array, time_array,
                      start_date_str, end_date_str, timeframe):
    """Save combined data to disk."""
    # Extract date parts for folder name
    start_short = start_date_str[2:10].replace('-', '')
    end_short = end_date_str[2:10].replace('-', '')

    data_folder = f'./data/trade_data/{timeframe}_{start_short}_{end_short}'
    os.makedirs(data_folder, exist_ok=True)

    print(f'\nSaving to {data_folder}...')

    with open(f'{data_folder}/data_from_processor', 'wb') as handle:
        pickle.dump(data_from_processor, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{data_folder}/price_array', 'wb') as handle:
        pickle.dump(price_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{data_folder}/tech_array', 'wb') as handle:
        pickle.dump(tech_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{data_folder}/time_array', 'wb') as handle:
        pickle.dump(time_array, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Saved successfully!')
    return data_folder


def main():
    # Configuration (from config_main.py)
    start_date = trade_start_date
    end_date = trade_end_date
    chunk_months = 3  # Download 3 months at a time

    print(f'\nChunked Data Download')
    print(f'{"="*80}')
    print(f'Start Date:  {start_date}')
    print(f'End Date:    {end_date}')
    print(f'Timeframe:   {TIMEFRAME}')
    print(f'Tickers:     {TICKER_LIST}')
    print(f'Chunk Size:  {chunk_months} months')
    print(f'{"="*80}\n')

    # Initialize processor
    source = DATA_SOURCE.lower()
    if source == 'coinbase':
        from processor_Coinbase import CoinbaseProcessor
        processor = CoinbaseProcessor()
    elif source == 'binance':
        from processor_Binance import BinanceProcessor
        processor = BinanceProcessor()
    else:
        raise ValueError(f"Unsupported DATA_SOURCE '{DATA_SOURCE}'")

    # Generate date ranges
    date_ranges = generate_date_ranges(start_date, end_date, chunk_months)
    print(f'Will download {len(date_ranges)} chunks:\n')
    for i, (s, e) in enumerate(date_ranges, 1):
        print(f'  Chunk {i}: {s} → {e}')
    print()

    # Download chunks
    chunks = []
    for i, (chunk_start, chunk_end) in enumerate(date_ranges, 1):
        print(f'\n[{i}/{len(date_ranges)}] Processing chunk {chunk_start} to {chunk_end}')

        result = download_chunk(
            processor, TICKER_LIST, chunk_start, chunk_end,
            TIMEFRAME, TECHNICAL_INDICATORS_LIST
        )

        chunks.append(result)

        # Small delay to avoid rate limits
        if i < len(date_ranges):
            print('  Waiting 2 seconds to avoid rate limits...')
            time.sleep(2)

    # Combine all chunks
    print('\n' + '='*80)
    combined_df, combined_price, combined_tech, combined_time = combine_chunks(chunks)

    # Save combined data
    data_folder = save_data_to_disk(
        combined_df, combined_price, combined_tech, combined_time,
        start_date, end_date, TIMEFRAME
    )

    print(f'\n{"="*80}')
    print('Download Complete!')
    print(f'Total Candles: {len(combined_time)}')
    print(f'Saved to: {data_folder}')
    print(f'{"="*80}\n')


if __name__ == "__main__":
    main()
