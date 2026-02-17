#!/usr/bin/env python3
"""
Convert 5m DataFrame to 30m Training Format

Resamples 5m data to 30m intervals (6 bars per 30m) and converts to
separate array files expected by the training script.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

print("Converting 5m → 30m training format...")
print()

# Load 5m DataFrame
with open('data/crypto_5m_6mo.pkl', 'rb') as f:
    df = pickle.load(f)

print(f"Loaded 5m DataFrame: {df.shape}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print()

# Filter to only the 7 tickers we're trading
DESIRED_TICKERS = ['AAVE/USD', 'AVAX/USD', 'BTC/USD', 'LINK/USD', 'ETH/USD', 'LTC/USD', 'UNI/USD']
print(f"Filtering to {len(DESIRED_TICKERS)} tickers: {DESIRED_TICKERS}")
df = df[df['tic'].isin(DESIRED_TICKERS)].copy()
print(f"Filtered DataFrame: {df.shape}")
print()

# Resample to 30m
print("Resampling 5m → 30m (6 bars per 30m interval)...")

# Set timestamp as index for resampling
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# Resample by ticker
resampled_dfs = []
for ticker in DESIRED_TICKERS:
    df_ticker = df[df['tic'] == ticker].copy()

    # Aggregation rules
    df_resampled = df_ticker.resample('30min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'vwap': 'mean',  # Volume-weighted average
        'trade_count': 'sum',
        'macd': 'last',  # Technical indicators: use last value
        'macd_signal': 'last',
        'macd_hist': 'last',
        'rsi': 'last',
        'cci': 'last',
        'dx': 'last'
    })

    # Drop rows with NaN values (from incomplete resampling periods)
    before_count = len(df_resampled)
    df_resampled = df_resampled.dropna()
    after_count = len(df_resampled)
    dropped = before_count - after_count

    # Add ticker column back
    df_resampled['tic'] = ticker
    df_resampled = df_resampled.reset_index()

    resampled_dfs.append(df_resampled)
    print(f"  ✓ {ticker}: {len(df_ticker)} → {after_count} bars ({len(df_ticker)/after_count:.1f}x compression, {dropped} NaN dropped)")


# Combine all tickers
df = pd.concat(resampled_dfs, ignore_index=True)
print(f"\nResampled DataFrame: {df.shape}")
print()

# Use fixed ticker count
tickers = len(DESIRED_TICKERS)
print(f"Using fixed ticker count: {tickers}")

# Check for timestamps with missing data
ticker_counts = df.groupby('timestamp').size()
incomplete_timestamps = ticker_counts[ticker_counts < tickers]
if len(incomplete_timestamps) > 0:
    print(f"⚠️  Found {len(incomplete_timestamps)} timestamps with incomplete data (< {tickers} tickers)")
    print(f"   These will be filtered out for data quality.")
    # Filter to only timestamps with all 7 tickers
    complete_timestamps = ticker_counts[ticker_counts == tickers].index
    df = df[df['timestamp'].isin(complete_timestamps)].copy()
    print(f"   Filtered to {len(complete_timestamps)} complete timestamps")
print()

# Price array: Use only closing price (to match 1h data format)
price_col = 'close'  # Single price per ticker

# Tech array: Must match 1h format with 14 values per ticker
tech_cols = ['open', 'high', 'low', 'close', 'volume', 'macd', 'macd_signal', 'macd_hist', 'rsi', 'cci', 'dx', 'vwap', 'trade_count']
# Total: 13. Will pad with 1 zero to match 1h format (14 per ticker)

# Get unique timestamps
timestamps = df['timestamp'].unique()
num_timestamps = len(timestamps)

print(f"Processing {num_timestamps} timestamps...")

# Initialize arrays
# price_array: One price (close) per ticker
price_array = np.zeros((num_timestamps, tickers))
# tech_array: 14 values per ticker (to match 1h format)
INDICATORS_PER_TICKER = 14
tech_array = np.zeros((num_timestamps, tickers * INDICATORS_PER_TICKER))
time_array = np.array([pd.Timestamp(ts).timestamp() for ts in timestamps])

# Fill arrays
for i, ts in enumerate(timestamps):
    if i % 1000 == 0:
        print(f"  Progress: {i}/{num_timestamps} ({i/num_timestamps*100:.1f}%)")

    df_ts = df[df['timestamp'] == ts].reset_index(drop=True)

    # Sort by ticker to ensure consistent order
    df_ts = df_ts.sort_values('tic').reset_index(drop=True)

    # Price array: [ticker1_close, ticker2_close, ticker3_close, ...]
    for j in range(len(df_ts)):  # Should be exactly 7 tickers due to filtering
        price_array[i, j] = df_ts.iloc[j][price_col]

    # Tech array: [ticker1_indicators(14), ticker2_indicators(14), ...]
    for j in range(len(df_ts)):
        # Write the 13 available indicators
        for k, col in enumerate(tech_cols):
            idx = j * INDICATORS_PER_TICKER + k
            tech_array[i, idx] = df_ts.iloc[j][col]
        # Pad with zero for the 14th indicator (to match 1h format)
        tech_array[i, j * INDICATORS_PER_TICKER + 13] = 0.0

print(f"  Progress: {num_timestamps}/{num_timestamps} (100.0%)")
print()

# Create output directory
output_dir = Path('data/30m')
output_dir.mkdir(exist_ok=True)

# Save arrays
print("Saving arrays...")
with open(output_dir / 'price_array', 'wb') as f:
    pickle.dump(price_array, f)
print(f"  ✓ Saved price_array: {price_array.shape}")

with open(output_dir / 'tech_array', 'wb') as f:
    pickle.dump(tech_array, f)
print(f"  ✓ Saved tech_array: {tech_array.shape}")

with open(output_dir / 'time_array', 'wb') as f:
    pickle.dump(time_array, f)
print(f"  ✓ Saved time_array: {time_array.shape}")

print()
print("✅ Conversion complete!")
print()
print("Training can now use: --data-dir data/30m")
