#!/usr/bin/env python3
"""
Download FRED Economic Data

This script downloads Federal Reserve economic data and prepares it for use
in the training pipeline. Run this before training to get latest macro indicators.

Usage:
    python 0_dl_fred_data.py
    python 0_dl_fred_data.py --timeframe 1h --lookback-days 730
"""

import argparse
import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path for imports
PARENT_DIR = Path(__file__).parent.parent / "ghost/FinRL_Crypto"
if PARENT_DIR.exists():
    sys.path.insert(0, str(PARENT_DIR))

from processor_FRED import FREDProcessor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description='Download FRED economic data')

    parser.add_argument(
        '--timeframe',
        type=str,
        default='1h',
        choices=['5m', '15m', '30m', '1h', '4h', '12h', '1d'],
        help='Timeframe to match crypto data (default: 1h)'
    )

    parser.add_argument(
        '--lookback-days',
        type=int,
        default=730,
        help='Number of days of historical data to download (default: 730 = 2 years)'
    )

    parser.add_argument(
        '--categories',
        nargs='+',
        default=['monetary', 'inflation', 'stress', 'fed_indices'],
        choices=['monetary', 'inflation', 'stress', 'fed_indices'],
        help='FRED data categories to download'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: data/{timeframe}_{bars})'
    )

    parser.add_argument(
        '--test-connection',
        action='store_true',
        help='Test FRED API connection and exit'
    )

    return parser.parse_args()


def test_fred_connection():
    """Test FRED API connection"""
    print("\n" + "="*80)
    print("TESTING FRED API CONNECTION")
    print("="*80 + "\n")

    try:
        processor = FREDProcessor()
        print("✓ FRED API key loaded successfully")

        # Try to download a single series
        print("\nTesting download (DFF - Fed Funds Rate)...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        processor.download_series(
            start_date=start_date,
            end_date=end_date,
            categories=['monetary']
        )

        if 'DFF' in processor.data and not processor.data['DFF'].empty:
            latest = processor.data['DFF'].iloc[-1]
            print(f"\n✓ Connection successful!")
            print(f"  Latest Fed Funds Rate: {latest['value']:.2f}%")
            print(f"  Date: {processor.data['DFF'].index[-1].strftime('%Y-%m-%d')}")
            return True
        else:
            print("\n✗ Connection failed - no data received")
            return False

    except ValueError as e:
        print(f"\n✗ API Key Error: {e}")
        print("\nPlease:")
        print("  1. Get free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("  2. Add to .env file: FRED_API_KEY=your_key_here")
        return False

    except Exception as e:
        print(f"\n✗ Connection Error: {e}")
        return False


def main():
    args = parse_args()

    # Test connection only
    if args.test_connection:
        success = test_fred_connection()
        sys.exit(0 if success else 1)

    print("\n" + "="*80)
    print("FRED DATA DOWNLOAD")
    print("="*80)
    print(f"Timeframe: {args.timeframe}")
    print(f"Lookback: {args.lookback_days} days")
    print(f"Categories: {', '.join(args.categories)}")
    print("="*80 + "\n")

    # Initialize processor
    try:
        processor = FREDProcessor()
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.lookback_days)

    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")

    # Step 1: Download FRED series
    processor.download_series(
        start_date=start_date,
        end_date=end_date,
        categories=args.categories
    )

    # Step 2: Resample to crypto timeframe
    print(f"\n{'='*80}")
    print("RESAMPLING TO CRYPTO TIMEFRAME")
    print(f"{'='*80}")
    resampled = processor.resample_to_timeframe(target_timeframe=args.timeframe)

    # Update processor data with resampled
    processor.data = resampled

    # Step 3: Compute features
    print(f"\n{'='*80}")
    print("COMPUTING FEATURES")
    print(f"{'='*80}")
    features_df = processor.compute_features(lookback_windows=[7, 30, 90])

    # Step 4: Save to disk
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Match crypto data directory structure
        # e.g., data/1h_1440 for 1h timeframe with 1440 bars
        # We don't know exact bar count yet, so use timeframe
        output_dir = f"data/{args.timeframe}_fred"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print("SAVING TO DISK")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")

    # Save raw FRED data
    processor.save_to_disk(output_dir)

    # Save features dataframe
    features_path = os.path.join(output_dir, 'macro_features_df.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump(features_df, f)
    print(f"  Saved features to {features_path}")

    # Save features as numpy array (for easy loading in training)
    array_path = os.path.join(output_dir, 'macro_features_array.pkl')
    with open(array_path, 'wb') as f:
        pickle.dump(features_df.values, f)
    print(f"  Saved array to {array_path}")

    # Save feature column names
    columns_path = os.path.join(output_dir, 'macro_feature_names.pkl')
    with open(columns_path, 'wb') as f:
        pickle.dump(features_df.columns.tolist(), f)
    print(f"  Saved column names to {columns_path}")

    # Step 5: Display summary
    print(f"\n{'='*80}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*80}")

    latest_values = processor.get_latest_values()

    print(f"\nLatest FRED Values ({latest_values['date'].max().strftime('%Y-%m-%d')}):")
    print("-" * 80)
    for _, row in latest_values.iterrows():
        print(f"{row['name']:40s} = {row['value']:12.4f}")

    print(f"\nFeatures created: {len(features_df.columns)}")
    print(f"Data points: {len(features_df)}")
    print(f"Date range: {features_df.index[0]} to {features_df.index[-1]}")
    print(f"Shape: {features_df.shape}")

    print(f"\n{'='*80}")
    print("✓ FRED DATA DOWNLOAD COMPLETE!")
    print(f"{'='*80}")
    print("\nNext steps:")
    print("  1. Run training with macro features:")
    print(f"     python 1_optimize_unified.py --use-macro --macro-dir {output_dir}")
    print("\n  2. Or update your existing training script to load macro features")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
