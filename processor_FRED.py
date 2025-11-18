#!/usr/bin/env python3
"""
FRED Data Processor for Macroeconomic Features

Downloads and processes Federal Reserve Economic Data (FRED) for use in
crypto trading models. Provides monetary policy, inflation, and market stress
indicators that drive crypto price action.

Author: Cappuccino Trading System
"""

import os
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fredapi import Fred

warnings.filterwarnings('ignore')


class FREDProcessor:
    """
    Downloads and processes FRED economic indicators for crypto prediction.

    Categories:
    - Monetary Policy: Fed Funds Rate, Balance Sheet, Reverse Repo
    - Inflation: CPI, Core CPI, PCE
    - Market Stress: Yield Curve, High Yield Spreads, Dollar Index
    - Fed Indices: Financial Conditions, Stress Indices
    """

    # Categorized FRED series with metadata
    MONETARY_POLICY = {
        'DFF': {
            'name': 'Fed Funds Rate',
            'units': 'Percent',
            'frequency': 'Daily',
            'impact': 'High ⬆️ = Crypto ⬇️'
        },
        'EFFR': {
            'name': 'Effective Fed Funds Rate',
            'units': 'Percent',
            'frequency': 'Daily',
            'impact': 'Real-time rate signal'
        },
        'WALCL': {
            'name': 'Fed Balance Sheet',
            'units': 'Millions of Dollars',
            'frequency': 'Weekly',
            'impact': 'High ⬆️ = Crypto ⬆️ (liquidity)'
        },
        'RRPONTSYD': {
            'name': 'Reverse Repo',
            'units': 'Billions of Dollars',
            'frequency': 'Daily',
            'impact': 'High = tight liquidity'
        },
    }

    INFLATION_INDICATORS = {
        'CPIAUCSL': {
            'name': 'Consumer Price Index',
            'units': 'Index 1982-1984=100',
            'frequency': 'Monthly',
            'impact': 'Inflation gauge'
        },
        'CPILFESL': {
            'name': 'Core CPI (ex food/energy)',
            'units': 'Index 1982-1984=100',
            'frequency': 'Monthly',
            'impact': "Fed's preferred metric"
        },
        'PCEPI': {
            'name': 'Personal Consumption Expenditures',
            'units': 'Index 2017=100',
            'frequency': 'Monthly',
            'impact': "Fed's inflation target"
        },
    }

    MARKET_STRESS = {
        'T10Y2Y': {
            'name': '10Y-2Y Treasury Spread',
            'units': 'Percent',
            'frequency': 'Daily',
            'impact': 'Negative = recession warning'
        },
        'BAMLH0A0HYM2': {
            'name': 'High Yield Spread',
            'units': 'Percent',
            'frequency': 'Daily',
            'impact': 'Credit stress indicator'
        },
        'DTWEXBGS': {
            'name': 'Trade-Weighted Dollar Index',
            'units': 'Index',
            'frequency': 'Daily',
            'impact': 'Strong dollar = weak crypto'
        },
        'VIXCLS': {
            'name': 'VIX Closing Price',
            'units': 'Index',
            'frequency': 'Daily',
            'impact': 'Market fear gauge'
        },
    }

    FED_STRESS_INDICES = {
        'NFCI': {
            'name': 'National Financial Conditions Index',
            'units': 'Index',
            'frequency': 'Weekly',
            'impact': 'Composite financial stress'
        },
        'STLFSI4': {
            'name': 'St. Louis Fed Financial Stress Index',
            'units': 'Index',
            'frequency': 'Weekly',
            'impact': 'Weekly stress gauge'
        },
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED client.

        Args:
            api_key: FRED API key. If None, reads from FRED_API_KEY env var.
                    Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html
        """
        if api_key is None:
            api_key = os.getenv('FRED_API_KEY')

        if not api_key or api_key == 'your_fred_api_key_here':
            raise ValueError(
                "FRED API key not found. Please:\n"
                "1. Get free API key at: https://fred.stlouisfed.org/docs/api/api_key.html\n"
                "2. Add to .env file: FRED_API_KEY=your_key_here"
            )

        self.fred = Fred(api_key=api_key)
        self.data: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, dict] = {}

        # Combine all series
        self.all_series = {}
        for category in [self.MONETARY_POLICY, self.INFLATION_INDICATORS,
                        self.MARKET_STRESS, self.FED_STRESS_INDICES]:
            self.all_series.update(category)

    def download_series(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        categories: List[str] = ['monetary', 'inflation', 'stress', 'fed_indices']
    ) -> Dict[str, pd.DataFrame]:
        """
        Download all FRED series in specified categories.

        Args:
            start_date: Start date for data (str 'YYYY-MM-DD' or datetime)
            end_date: End date for data
            categories: List of categories to download:
                       'monetary', 'inflation', 'stress', 'fed_indices'

        Returns:
            Dictionary of DataFrames keyed by series ID
        """
        # Convert dates to strings if needed
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')

        # Build series dict based on categories
        series_dict = {}
        if 'monetary' in categories:
            series_dict.update(self.MONETARY_POLICY)
        if 'inflation' in categories:
            series_dict.update(self.INFLATION_INDICATORS)
        if 'stress' in categories:
            series_dict.update(self.MARKET_STRESS)
        if 'fed_indices' in categories:
            series_dict.update(self.FED_STRESS_INDICES)

        print(f"\n{'='*80}")
        print(f"DOWNLOADING FRED ECONOMIC DATA")
        print(f"{'='*80}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Categories: {', '.join(categories)}")
        print(f"Series count: {len(series_dict)}")
        print(f"{'='*80}\n")

        for series_id, metadata in series_dict.items():
            print(f"Downloading {metadata['name']} ({series_id})...")
            try:
                data = self.fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date
                )

                self.data[series_id] = pd.DataFrame({
                    'value': data,
                })
                self.metadata[series_id] = metadata

                print(f"  ✓ Downloaded {len(data)} observations")
                print(f"    Latest: {data.index[-1].strftime('%Y-%m-%d')} = {data.iloc[-1]:.4f}")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                # Create empty dataframe as placeholder
                self.data[series_id] = pd.DataFrame()

        print(f"\n{'='*80}")
        print(f"DOWNLOAD COMPLETE: {len([d for d in self.data.values() if not d.empty])}/{len(series_dict)} series")
        print(f"{'='*80}\n")

        return self.data

    def resample_to_timeframe(
        self,
        target_timeframe: str = '1h',
        method: str = 'ffill'
    ) -> Dict[str, pd.DataFrame]:
        """
        Resample FRED data to match crypto timeframe.

        FRED data is daily/weekly/monthly, but we need hourly/5min data.
        Uses forward-fill by default: CPI is monthly, but we need value for
        every hour → use last known value until new data arrives.

        Args:
            target_timeframe: Target timeframe ('5m', '1h', '4h', '1d')
            method: Resampling method ('ffill', 'bfill', 'interpolate')

        Returns:
            Dictionary of resampled DataFrames
        """
        resampled = {}

        print(f"\nResampling to {target_timeframe} timeframe using {method}...")

        for series_id, df in self.data.items():
            if df.empty:
                resampled[series_id] = df
                continue

            # Resample to target frequency
            if method == 'ffill':
                resampled_df = df.resample(target_timeframe).ffill()
            elif method == 'bfill':
                resampled_df = df.resample(target_timeframe).bfill()
            elif method == 'interpolate':
                resampled_df = df.resample(target_timeframe).interpolate()
            else:
                raise ValueError(f"Unknown method: {method}")

            resampled[series_id] = resampled_df
            print(f"  {series_id}: {len(df)} → {len(resampled_df)} bars")

        return resampled

    def compute_features(
        self,
        lookback_windows: List[int] = [7, 30, 90]
    ) -> pd.DataFrame:
        """
        Compute derived features from FRED series.

        Creates:
        - Level features (current value)
        - Rate of change (1-month, 3-month)
        - Z-scores (statistical deviation)
        - Composite features (real rates, liquidity proxies)

        Args:
            lookback_windows: Windows for rolling calculations (in days)

        Returns:
            DataFrame with all macro features aligned to same index
        """
        print(f"\nComputing macro features...")

        # Find common index (union of all series)
        all_indices = [df.index for df in self.data.values() if not df.empty]
        if not all_indices:
            return pd.DataFrame()

        common_index = all_indices[0]
        for idx in all_indices[1:]:
            common_index = common_index.union(idx)

        features = pd.DataFrame(index=common_index)

        # Compute features for each series
        for series_id, df in self.data.items():
            if df.empty:
                continue

            # Reindex to common index (forward fill)
            df_aligned = df.reindex(common_index).ffill()

            # 1. Level
            features[f'{series_id}'] = df_aligned['value']

            # 2. Rate of change (percentage)
            for window in lookback_windows:
                features[f'{series_id}_chg_{window}d'] = df_aligned['value'].pct_change(window) * 100

            # 3. Z-score (how extreme is current value?)
            for window in [90, 180]:
                rolling_mean = df_aligned['value'].rolling(window).mean()
                rolling_std = df_aligned['value'].rolling(window).std()
                features[f'{series_id}_zscore_{window}d'] = (
                    (df_aligned['value'] - rolling_mean) / rolling_std
                )

        # Composite features
        print("  Computing composite features...")

        # Real interest rate (nominal rate - inflation)
        if 'DFF' in self.data and 'CPIAUCSL' in self.data and not self.data['DFF'].empty and not self.data['CPIAUCSL'].empty:
            inflation_rate = self.data['CPIAUCSL']['value'].pct_change(12) * 100  # YoY inflation
            inflation_aligned = inflation_rate.reindex(common_index).ffill()
            fed_funds_aligned = self.data['DFF']['value'].reindex(common_index).ffill()
            features['real_rate'] = fed_funds_aligned - inflation_aligned
            print("    ✓ Real interest rate")

        # Yield curve inversion signal
        if 'T10Y2Y' in self.data and not self.data['T10Y2Y'].empty:
            yield_curve = self.data['T10Y2Y']['value'].reindex(common_index).ffill()
            features['yield_curve_inverted'] = (yield_curve < 0).astype(float)
            features['yield_curve_inversion_depth'] = np.minimum(yield_curve, 0)  # How negative
            print("    ✓ Yield curve signals")

        # Fed balance sheet expansion rate
        if 'WALCL' in self.data and not self.data['WALCL'].empty:
            bs = self.data['WALCL']['value'].reindex(common_index).ffill()
            features['fed_bs_expansion_1m'] = bs.pct_change(30) * 100
            features['fed_bs_expansion_3m'] = bs.pct_change(90) * 100
            print("    ✓ Fed balance sheet expansion")

        # Liquidity stress (Reverse Repo as % of Fed Balance Sheet)
        if 'RRPONTSYD' in self.data and 'WALCL' in self.data:
            if not self.data['RRPONTSYD'].empty and not self.data['WALCL'].empty:
                rrp = self.data['RRPONTSYD']['value'].reindex(common_index).ffill()
                bs = self.data['WALCL']['value'].reindex(common_index).ffill()
                features['liquidity_stress'] = (rrp * 1000) / bs * 100  # RRP in billions, BS in millions
                print("    ✓ Liquidity stress ratio")

        # Market fear composite (VIX + High Yield Spread)
        if 'VIXCLS' in self.data and 'BAMLH0A0HYM2' in self.data:
            if not self.data['VIXCLS'].empty and not self.data['BAMLH0A0HYM2'].empty:
                vix = self.data['VIXCLS']['value'].reindex(common_index).ffill()
                hy_spread = self.data['BAMLH0A0HYM2']['value'].reindex(common_index).ffill()

                # Normalize and combine
                vix_norm = (vix - vix.rolling(180).mean()) / vix.rolling(180).std()
                hy_norm = (hy_spread - hy_spread.rolling(180).mean()) / hy_spread.rolling(180).std()

                features['market_fear_composite'] = (vix_norm + hy_norm) / 2
                print("    ✓ Market fear composite")

        print(f"\n  Total features created: {len(features.columns)}")
        print(f"  Date range: {features.index[0]} to {features.index[-1]}")
        print(f"  Shape: {features.shape}")

        return features

    def align_to_crypto_data(
        self,
        crypto_index: pd.DatetimeIndex,
        features_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Align FRED features to crypto data index and convert to numpy array.

        Args:
            crypto_index: DatetimeIndex from crypto data
            features_df: DataFrame with FRED features

        Returns:
            numpy array (n_bars, n_features) aligned to crypto data
        """
        print(f"\nAligning FRED features to crypto data...")
        print(f"  Crypto bars: {len(crypto_index)}")
        print(f"  FRED features: {len(features_df.columns)}")

        # Reindex to crypto timestamps (forward fill)
        aligned = features_df.reindex(crypto_index, method='ffill')

        # Fill any remaining NaNs with 0
        aligned = aligned.fillna(0)

        # Convert to numpy
        array = aligned.values

        print(f"  Output shape: {array.shape}")
        print(f"  NaN count: {np.isnan(array).sum()}")

        return array

    def get_latest_values(self) -> pd.DataFrame:
        """
        Get current values of all indicators.

        Returns:
            DataFrame with latest value, date, and description for each series
        """
        latest_data = []

        for series_id, df in self.data.items():
            if df.empty:
                continue

            latest_data.append({
                'series_id': series_id,
                'name': self.metadata[series_id]['name'],
                'value': df['value'].iloc[-1],
                'date': df.index[-1],
                'units': self.metadata[series_id]['units'],
                'impact': self.metadata[series_id]['impact']
            })

        return pd.DataFrame(latest_data)

    def save_to_disk(self, output_dir: str):
        """
        Save FRED data and features to disk.

        Args:
            output_dir: Directory to save files (e.g., 'data/1h_1440')
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save raw data
        raw_path = os.path.join(output_dir, 'fred_raw_data.pkl')
        with open(raw_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"  Saved raw data to {raw_path}")

        # Save metadata
        meta_path = os.path.join(output_dir, 'fred_metadata.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"  Saved metadata to {meta_path}")

    @staticmethod
    def load_from_disk(input_dir: str):
        """
        Load FRED data from disk.

        Args:
            input_dir: Directory containing saved files

        Returns:
            Tuple of (data_dict, metadata_dict)
        """
        raw_path = os.path.join(input_dir, 'fred_raw_data.pkl')
        meta_path = os.path.join(input_dir, 'fred_metadata.pkl')

        with open(raw_path, 'rb') as f:
            data = pickle.load(f)

        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)

        return data, metadata


def main():
    """Example usage"""
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Initialize processor
    processor = FREDProcessor()

    # Download data (2 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    processor.download_series(
        start_date=start_date,
        end_date=end_date,
        categories=['monetary', 'inflation', 'stress', 'fed_indices']
    )

    # Compute features
    features_df = processor.compute_features()

    # Display latest values
    print("\nLatest FRED Values:")
    print("=" * 80)
    latest = processor.get_latest_values()
    for _, row in latest.iterrows():
        print(f"{row['name']:40s} = {row['value']:12.4f} ({row['date'].strftime('%Y-%m-%d')})")
        print(f"  Impact: {row['impact']}")

    print("\nDone! Use this data in your training pipeline.")


if __name__ == "__main__":
    main()
