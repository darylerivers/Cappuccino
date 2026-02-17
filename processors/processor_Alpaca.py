"""Alpaca processor for downloading crypto historical data."""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import TimeFrame
except ImportError:
    raise ImportError("alpaca-trade-api is required. Run: pip install alpaca-trade-api")

from talib import RSI, MACD, CCI, DX, ATR, ADX

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required if env vars already set

# Load API credentials from environment
API_KEY_ALPACA = os.getenv("ALPACA_API_KEY", "")
API_SECRET_ALPACA = os.getenv("ALPACA_API_SECRET", "")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() in {"1", "true", "yes"}


class AlpacaProcessor:
    def __init__(self):
        self.end_date = None
        self.start_date = None
        self.tech_indicator_list = None
        self.api_key = API_KEY_ALPACA
        self.api_secret = API_SECRET_ALPACA

        if not self.api_key or not self.api_secret:
            raise ValueError("ALPACA_API_KEY and ALPACA_API_SECRET environment variables required")

        base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
        self.api = tradeapi.REST(self.api_key, self.api_secret, base_url, api_version='v2')
        self.ticker_list = []

    def run(self, ticker_list, start_date, end_date, time_interval, technical_indicator_list, if_vix):
        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

        print('Downloading data from Alpaca...')
        data = self.download_data(ticker_list, start_date, end_date, time_interval)
        print('Downloading finished! Transforming data...')
        data = self.clean_data(data)
        data = self.add_technical_indicator(data, technical_indicator_list)

        price_array, tech_array, time_array = self.df_to_array(data, if_vix)

        # Handle NaN values
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0

        return data, price_array, tech_array, time_array

    def download_data(self, ticker_list, start_date, end_date, time_interval):
        """Download historical bars from Alpaca."""
        # Map timeframe string to Alpaca TimeFrame
        timeframe_map = {
            '1m': TimeFrame.Minute,
            '5m': TimeFrame(5, 'Min'),
            '15m': TimeFrame(15, 'Min'),
            '30m': TimeFrame(30, 'Min'),
            '1h': TimeFrame.Hour,
            '4h': TimeFrame(4, 'Hour'),
            '1d': TimeFrame.Day,
        }

        if time_interval not in timeframe_map:
            raise ValueError(f"Unsupported timeframe: {time_interval}")

        tf = timeframe_map[time_interval]

        # Parse dates and format for Alpaca (YYYY-MM-DD format only)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")

        # Format as YYYY-MM-DD for Alpaca
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")

        final_df = pd.DataFrame()

        for ticker in ticker_list:
            print(f"  Downloading {ticker}...")
            try:
                # Alpaca expects tickers with slash (e.g., "BTC/USD")
                symbol = ticker if '/' in ticker else f"{ticker}/USD"
                bars = self.api.get_crypto_bars(
                    symbol,
                    tf,
                    start=start_str,
                    end=end_str,
                ).df

                if bars.empty:
                    print(f"  WARNING: No data received for {ticker}")
                    continue

                bars = bars.reset_index()
                bars['tic'] = ticker
                bars = bars.rename(columns={
                    'timestamp': 'time',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume',
                })

                # Keep only required columns
                bars = bars[['time', 'tic', 'open', 'high', 'low', 'close', 'volume']]
                final_df = pd.concat([final_df, bars], ignore_index=True)

            except Exception as e:
                print(f"  ERROR downloading {ticker}: {e}")
                continue

        return final_df

    def clean_data(self, df):
        """Clean the data."""
        df = df.dropna()
        df = df.sort_values(by=['time', 'tic'])
        df = df.reset_index(drop=True)
        return df

    def add_technical_indicator(self, df, tech_indicator_list):
        """Add technical indicators using TA-Lib."""
        final_df = pd.DataFrame()

        for ticker in df.tic.unique():
            tic_df = df[df.tic == ticker].copy()

            # Existing indicators
            tic_df['macd'], tic_df['macd_signal'], tic_df['macd_hist'] = MACD(
                tic_df['close'], fastperiod=12, slowperiod=26, signalperiod=9
            )
            tic_df['rsi'] = RSI(tic_df['close'], timeperiod=14)
            tic_df['cci'] = CCI(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
            tic_df['dx'] = DX(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)

            # NEW FEATURE 1: ATR regime shift
            # Detects volatility regime changes by comparing current ATR to historical average
            atr_14 = ATR(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
            atr_50_ma = atr_14.rolling(window=50, min_periods=1).mean()
            tic_df['atr_regime_shift'] = (atr_14 - atr_50_ma) / (atr_50_ma + 1e-8)  # Normalized shift

            # NEW FEATURE 2: Range breakout + volume
            # Detects price breakouts from recent range with volume confirmation
            range_period = 20
            recent_high = tic_df['high'].rolling(window=range_period, min_periods=1).max()
            recent_low = tic_df['low'].rolling(window=range_period, min_periods=1).min()
            range_size = recent_high - recent_low

            # Breakout signal: 1 if breaking above high, -1 if breaking below low, 0 otherwise
            breakout_signal = np.where(tic_df['close'] > recent_high, 1.0,
                                     np.where(tic_df['close'] < recent_low, -1.0, 0.0))

            # Volume confirmation: compare to 20-period average volume
            avg_volume = tic_df['volume'].rolling(window=20, min_periods=1).mean()
            volume_ratio = tic_df['volume'] / (avg_volume + 1e-8)

            # Combined signal: breakout * volume ratio (higher when breakout occurs with high volume)
            tic_df['range_breakout_volume'] = breakout_signal * volume_ratio

            # NEW FEATURE 3: Trend strength re-acceleration
            # Detects when trend strength (ADX) is accelerating (second derivative)
            adx_14 = ADX(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
            adx_change = adx_14.diff(periods=1)  # First derivative (momentum)
            adx_acceleration = adx_change.diff(periods=1)  # Second derivative (acceleration)
            tic_df['trend_reacceleration'] = adx_acceleration

            final_df = pd.concat([final_df, tic_df], ignore_index=True)

        # Drop rows with NaN (from indicator calculation)
        final_df = final_df.dropna()
        final_df = final_df.sort_values(by=['time', 'tic'])
        final_df = final_df.reset_index(drop=True)

        print(f"Added technical indicators. Final shape: {final_df.shape}")
        return final_df

    def df_to_array(self, df, if_vix):
        """Convert dataframe to arrays."""
        self.tech_indicator_list = list(df.columns)
        self.tech_indicator_list.remove('tic')
        self.tech_indicator_list.remove('time')

        print(f'Converting to arrays with {len(self.tech_indicator_list)} technical indicators: {self.tech_indicator_list}')

        unique_ticker = df.tic.unique()
        if_first_time = True

        for tic in unique_ticker:
            tic_data = df[df.tic == tic]
            if if_first_time:
                price_array = tic_data[['close']].values
                tech_array = tic_data[self.tech_indicator_list].values
                time_array = tic_data['time'].values
                if_first_time = False
            else:
                price_array = np.hstack([price_array, tic_data[['close']].values])
                tech_array = np.hstack([tech_array, tic_data[self.tech_indicator_list].values])

        assert price_array.shape[0] == tech_array.shape[0], f"Shape mismatch: price {price_array.shape} vs tech {tech_array.shape}"

        print(f"Arrays created - price: {price_array.shape}, tech: {tech_array.shape}, time: {len(time_array)}")
        return price_array, tech_array, time_array
