import yfinance as yf
import pandas as pd
import numpy as np

class MultiAssetProcessor:
    TIER1_ASSETS = {
        'SPY': 'S&P 500',
        'QQQ': 'Nasdaq 100',
        'DXY': 'US Dollar Index',
        'GLD': 'Gold',
        'VIX': 'Volatility Index'
    }

    TIER2_ASSETS = {
        'MSTR': 'MicroStrategy',
        'COIN': 'Coinbase',
        'TSLA': 'Tesla',
        'NVDA': 'NVIDIA'
    }

    def __init__(self, start_date, end_date, timeframe='1h'):
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.data = {}

    def download_all_assets(self, tiers=['tier1']):
        assets = {}
        if 'tier1' in tiers:
            assets.update(self.TIER1_ASSETS)
        if 'tier2' in tiers:
            assets.update(self.TIER2_ASSETS)
        
        for ticker, name in assets.items():
            print(f"Downloading {name} ({ticker})...")
            try:
                self.data[ticker] = yf.download(
                    ticker,
                    start = self.start_date,
                    end = self.end_date,
                    interval = self.timeframe,
                    progress = False
                )
            except Exception as e:
                print(f" Error downloading {ticker}: {e}")
        return self.data
    
    def compute_correlation_features(self, crypto_df, window=30):

        features = pd.DataFrame(index = crypto_df.index)
        crypto_returns = crypto_df['close'].pct_change()

        for ticker, df in self.data.items():
            if df.empty:
                continue
            asset_returns = df['Close'].pct_change()

            features[f'{ticker}_corr_{window}d'] = crypto_returns.rolling(window).corr(asset_returns)

            features[f'{ticker}_corr_trend'] = features[f'{ticker}_corr_{window}d'].diff(7)

            # Price Ratio
            features[f'{ticker}_ratio'] = crypto_df['close'] / df['Close']
            features[f'{ticker}_ratio_ma'] = features[f'{ticker}_ratio'].rolling(30).mean()
            features[f'{ticker}_ratio_std'] = features[f'{ticker}_ratio'].rolling(30).std()
        return features
    
    def compute_divergence_signals(self, crypto_df):
        signals = pd.DataFrame(index=crypto_df.index)

        crypto_trend = crypto_df('close').rolling(7).mean().diff()

        for ticker, df in self.data.items():
            if df.empty or ticker == 'VIX':
                continue
            asset_trend = df['Close'].rolling(7).mean().diff()

            signals[f'{ticker}_divergence'] = np.sign(crypto_trend) != np.sign(asset_trend)
            signals[f'{ticker}_divergence'] = signals[f'{ticker}_divergence'].astype(float)
        return signals
