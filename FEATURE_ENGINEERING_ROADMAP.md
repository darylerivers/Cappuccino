# Feature Engineering Roadmap for Improved Model Accuracy

## Executive Summary

This document outlines a comprehensive strategy to enhance the predictive accuracy of the Cappuccino trading model through three main enhancements:

1. **Multi-Asset Correlation Features** - Leverage traditional markets and crypto indices
2. **Macroeconomic Indicators via FRED API** - Federal Reserve and government economic data
3. **Fed-Focused Sentiment Analysis** - FOMC statements, Fed research, and policy signals

---

## Current State Analysis

### Existing Features (per ticker)
- **Price data**: open, high, low, close, volume (5 features)
- **Technical indicators**: MACD (3), RSI (1), CCI (1), DX (1) = 6 features
- **Sentiment**: Fear & Greed Index (1 feature per market)
- **Total**: ~12 features per ticker

### State Dimension
With 3 tickers (BTC, ETH, SOL) and lookback=20:
- State dim = 1 (cash) + 3 (stocks) + 12 * 3 * 20 = ~724 dimensions

### Limitations of Current Approach
1. **Crypto-only focus** - Ignores broader market context
2. **Technical-only indicators** - Lacks fundamental/macro signals
3. **Generic sentiment** - Not crypto-specific or macro-aware
4. **No cross-asset signals** - Missing correlation/divergence signals

---

## Enhancement 1: Multi-Asset Correlation Features

### Rationale
Crypto markets don't exist in isolation. They correlate with:
- Traditional equity indices (risk-on/risk-off sentiment)
- Dollar strength (DXY - inverse correlation with crypto)
- Volatility indices (VIX - fear gauge)
- Gold (competing safe haven)
- Tech stocks (sector correlation)

### Recommended Assets to Track

#### **Tier 1: Critical Correlations (High Priority)**

| Asset | Ticker | Relationship | Why It Matters |
|-------|--------|--------------|----------------|
| **S&P 500** | SPY | +0.6 to +0.8 correlation | Crypto follows equities in risk-on/off |
| **Nasdaq 100** | QQQ | +0.7 to +0.85 correlation | Tech correlation, liquidity proxy |
| **US Dollar Index** | DXY | -0.5 to -0.7 correlation | Strong dollar = weak crypto |
| **Gold** | GLD | +0.3 to +0.5 correlation | Competing store of value |
| **VIX (Volatility)** | VIX | -0.4 to -0.6 correlation | Fear gauge, risk appetite |

#### **Tier 2: Sector/Thematic Correlations**

| Asset | Ticker | Relationship | Why It Matters |
|-------|--------|--------------|----------------|
| **Bitcoin Dominance** | BTC.D | N/A | Alt season indicator |
| **Total Crypto Market Cap** | TOTAL | +0.95+ correlation | Overall market health |
| **MicroStrategy** | MSTR | +0.85+ correlation | Bitcoin proxy stock |
| **Coinbase** | COIN | +0.75+ correlation | Crypto exchange proxy |
| **Tech Giants** | TSLA, NVDA | +0.4 to +0.6 | Liquidity/tech correlation |

#### **Tier 3: Macro Indicators**

| Asset | Ticker | Relationship | Why It Matters |
|-------|--------|--------------|----------------|
| **10Y Treasury Yield** | ^TNX | -0.3 to -0.5 correlation | Discount rate for risk assets |
| **2Y Treasury Yield** | ^IRX | -0.2 to -0.4 correlation | Near-term rate expectations |
| **Oil** | USO | +0.2 to +0.4 correlation | Inflation/energy proxy |
| **High Yield Bonds** | HYG | +0.5 to +0.7 correlation | Credit risk appetite |

### Features to Extract

For each correlated asset, compute:

1. **Price-based features**:
   - Returns (1d, 7d, 30d)
   - Volatility (rolling 7d, 30d)
   - Z-score (price deviation from mean)

2. **Correlation features**:
   - Rolling correlation (7d, 30d) with BTC/ETH
   - Correlation rank (is it strengthening/weakening?)
   - Divergence signal (crypto up, equity down = warning)

3. **Relative strength**:
   - Crypto/SPY ratio
   - Crypto/Gold ratio
   - BTC Dominance changes

### Implementation Strategy

```python
# processor_MultiAsset.py - NEW FILE

import yfinance as yf
import pandas as pd
import numpy as np

class MultiAssetProcessor:
    """
    Downloads and processes correlated traditional assets for crypto prediction.
    """

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
        """Download all correlated assets."""
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
                    start=self.start_date,
                    end=self.end_date,
                    interval=self.timeframe,
                    progress=False
                )
            except Exception as e:
                print(f"  Error downloading {ticker}: {e}")

        return self.data

    def compute_correlation_features(self, crypto_df, window=30):
        """
        Compute rolling correlations between crypto and traditional assets.

        Args:
            crypto_df: DataFrame with crypto OHLCV data
            window: Rolling window for correlation (in bars)

        Returns:
            DataFrame with correlation features
        """
        features = pd.DataFrame(index=crypto_df.index)

        crypto_returns = crypto_df['close'].pct_change()

        for ticker, df in self.data.items():
            if df.empty:
                continue

            # Resample to match crypto timeframe
            asset_returns = df['Close'].pct_change()

            # Rolling correlation
            features[f'{ticker}_corr_{window}d'] = crypto_returns.rolling(window).corr(asset_returns)

            # Correlation trend (is it increasing?)
            features[f'{ticker}_corr_trend'] = features[f'{ticker}_corr_{window}d'].diff(7)

            # Price ratio (crypto/asset)
            features[f'{ticker}_ratio'] = crypto_df['close'] / df['Close']
            features[f'{ticker}_ratio_ma'] = features[f'{ticker}_ratio'].rolling(30).mean()
            features[f'{ticker}_ratio_std'] = features[f'{ticker}_ratio'].rolling(30).std()

        return features

    def compute_divergence_signals(self, crypto_df):
        """
        Detect divergences between crypto and traditional markets.

        Returns:
            DataFrame with divergence signals
        """
        signals = pd.DataFrame(index=crypto_df.index)

        crypto_trend = crypto_df['close'].rolling(7).mean().diff()

        for ticker, df in self.data.items():
            if df.empty or ticker == 'VIX':  # VIX is special
                continue

            asset_trend = df['Close'].rolling(7).mean().diff()

            # Divergence: crypto up, market down (bearish for crypto)
            signals[f'{ticker}_divergence'] = np.sign(crypto_trend) != np.sign(asset_trend)
            signals[f'{ticker}_divergence'] = signals[f'{ticker}_divergence'].astype(float)

        return signals
```

### Expected Impact
- **+5-10% accuracy improvement** from market context
- **Better risk-off detection** - VIX spike = get out
- **Clearer trend confirmation** - crypto + equities both up = strong signal

---

## Enhancement 2: FRED API for Macroeconomic Indicators

### Rationale
Crypto is increasingly driven by macro factors:
- **Interest rates** - Higher rates = lower crypto prices
- **Liquidity** - Fed balance sheet expansion = crypto rallies
- **Inflation** - CPI prints move crypto markets
- **Dollar strength** - Inverse correlation with crypto

### FRED API Setup

```bash
# Install FRED API client
pip install fredapi

# Get API key (free)
# Visit: https://fred.stlouisfed.org/docs/api/api_key.html
```

### Recommended FRED Series

#### **Tier 1: Monetary Policy Indicators**

| Series ID | Description | Update Frequency | Impact on Crypto |
|-----------|-------------|------------------|------------------|
| **DFF** | Federal Funds Rate | Daily | High ⬆️ = Crypto ⬇️ |
| **EFFR** | Effective Federal Funds Rate | Daily | Real-time rate signal |
| **WALCL** | Fed Balance Sheet | Weekly | High ⬆️ = Crypto ⬆️ (liquidity) |
| **RRPONTSYD** | Reverse Repo | Daily | High = tight liquidity |
| **TERMCBCCALLNS** | Central Bank Liquidity Swaps | Weekly | Crisis indicator |

#### **Tier 2: Inflation & Economic Health**

| Series ID | Description | Update Frequency | Impact |
|-----------|-------------|------------------|--------|
| **CPIAUCSL** | Consumer Price Index | Monthly | Inflation gauge |
| **CPILFESL** | Core CPI (ex food/energy) | Monthly | Fed's preferred metric |
| **PCEPI** | Personal Consumption Expenditures | Monthly | Fed's inflation target |
| **UNRATE** | Unemployment Rate | Monthly | Economic health |
| **UMCSENT** | U Michigan Consumer Sentiment | Monthly | Recession early warning |

#### **Tier 3: Market Stress & Liquidity**

| Series ID | Description | Update Frequency | Impact |
|-----------|-------------|------------------|--------|
| **T10Y2Y** | 10Y-2Y Treasury Spread | Daily | Recession predictor |
| **BAMLH0A0HYM2** | High Yield Spread | Daily | Credit stress |
| **DTWEXBGS** | Trade-Weighted Dollar Index | Daily | Dollar strength |
| **VIXCLS** | VIX Closing Price | Daily | Market fear |

#### **Tier 4: Fed-Specific Research Indicators**

| Series ID | Description | Update Frequency | Impact |
|-----------|-------------|------------------|--------|
| **NFCI** | National Financial Conditions Index | Weekly | Financial stress composite |
| **STLFSI4** | St. Louis Fed Financial Stress Index | Weekly | Stress gauge |
| **ANFCI** | Adjusted NFCI | Weekly | Adjusted for economic growth |

### Implementation

```python
# processor_FRED.py - NEW FILE

from fredapi import Fred
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FREDProcessor:
    """
    Downloads and processes FRED economic indicators for crypto prediction.
    """

    # Categorized series for easy management
    MONETARY_POLICY = {
        'DFF': 'Fed Funds Rate',
        'EFFR': 'Effective Fed Funds Rate',
        'WALCL': 'Fed Balance Sheet',
        'RRPONTSYD': 'Reverse Repo',
    }

    INFLATION_INDICATORS = {
        'CPIAUCSL': 'CPI',
        'CPILFESL': 'Core CPI',
        'PCEPI': 'PCE Index',
    }

    MARKET_STRESS = {
        'T10Y2Y': '10Y-2Y Spread',
        'BAMLH0A0HYM2': 'High Yield Spread',
        'DTWEXBGS': 'Dollar Index',
        'VIXCLS': 'VIX',
    }

    FED_STRESS_INDICES = {
        'NFCI': 'Financial Conditions Index',
        'STLFSI4': 'Financial Stress Index',
        'ANFCI': 'Adjusted NFCI',
    }

    def __init__(self, api_key):
        """
        Initialize FRED client.

        Args:
            api_key: FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html
        """
        self.fred = Fred(api_key=api_key)
        self.data = {}

    def download_series(self, start_date, end_date, categories=['monetary', 'inflation', 'stress']):
        """
        Download all FRED series in specified categories.

        Args:
            start_date: Start date for data
            end_date: End date for data
            categories: List of categories to download

        Returns:
            Dictionary of DataFrames keyed by series ID
        """
        series_dict = {}

        if 'monetary' in categories:
            series_dict.update(self.MONETARY_POLICY)
        if 'inflation' in categories:
            series_dict.update(self.INFLATION_INDICATORS)
        if 'stress' in categories:
            series_dict.update(self.MARKET_STRESS)
        if 'fed_indices' in categories:
            series_dict.update(self.FED_STRESS_INDICES)

        for series_id, description in series_dict.items():
            print(f"Downloading {description} ({series_id})...")
            try:
                data = self.fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date
                )
                self.data[series_id] = pd.DataFrame({
                    'value': data,
                    'series_id': series_id,
                    'description': description
                })
                print(f"  ✓ Downloaded {len(data)} observations")
            except Exception as e:
                print(f"  ✗ Error: {e}")

        return self.data

    def resample_to_crypto_timeframe(self, crypto_timeframe='1h'):
        """
        Resample FRED data (daily/weekly/monthly) to crypto timeframe.

        Uses forward-fill for low-frequency indicators (CPI updates monthly,
        but we need values for every hour).

        Args:
            crypto_timeframe: Target timeframe ('5m', '1h', '4h', '1d')

        Returns:
            Dictionary of resampled series
        """
        resampled = {}

        for series_id, df in self.data.items():
            # Forward-fill: use last known value until new data arrives
            resampled[series_id] = df.resample(crypto_timeframe).ffill()

        return resampled

    def compute_macro_features(self):
        """
        Compute derived features from FRED series.

        Returns:
            DataFrame with macro features
        """
        features = pd.DataFrame()

        # Rate of change features
        for series_id, df in self.data.items():
            if df.empty:
                continue

            # Level
            features[f'{series_id}'] = df['value']

            # Rate of change (1-month, 3-month, 6-month)
            features[f'{series_id}_chg_1m'] = df['value'].pct_change(30)
            features[f'{series_id}_chg_3m'] = df['value'].pct_change(90)

            # Z-score (how extreme is current value?)
            rolling_mean = df['value'].rolling(180).mean()
            rolling_std = df['value'].rolling(180).std()
            features[f'{series_id}_zscore'] = (df['value'] - rolling_mean) / rolling_std

        # Composite features
        if 'DFF' in self.data and 'CPIAUCSL' in self.data:
            # Real interest rate (nominal rate - inflation)
            features['real_rate'] = self.data['DFF']['value'] - self.data['CPIAUCSL']['value'].pct_change(12) * 100

        if 'T10Y2Y' in self.data:
            # Yield curve inversion signal
            features['yield_curve_inverted'] = (self.data['T10Y2Y']['value'] < 0).astype(float)

        if 'WALCL' in self.data:
            # Fed balance sheet expansion rate
            features['fed_bs_expansion'] = self.data['WALCL']['value'].pct_change(30)

        return features

    def get_latest_values(self):
        """Get current values of all indicators."""
        latest = {}
        for series_id, df in self.data.items():
            if not df.empty:
                latest[series_id] = {
                    'value': df['value'].iloc[-1],
                    'date': df.index[-1],
                    'description': df['description'].iloc[0]
                }
        return latest
```

### Usage Example

```python
from processor_FRED import FREDProcessor
from datetime import datetime, timedelta
import os

# Initialize
fred_api_key = os.getenv('FRED_API_KEY')  # Store in .env
fred = FREDProcessor(api_key=fred_api_key)

# Download data
end_date = datetime.now()
start_date = end_date - timedelta(days=730)  # 2 years

fred.download_series(
    start_date=start_date.strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d'),
    categories=['monetary', 'inflation', 'stress', 'fed_indices']
)

# Resample to 1h to match crypto data
resampled = fred.resample_to_crypto_timeframe('1h')

# Compute features
macro_features = fred.compute_macro_features()

# Save for use in training
import pickle
with open('data/1h_1440/macro_features', 'wb') as f:
    pickle.dump(macro_features, f)
```

### Environment Integration

Update `environment_Alpaca.py`:

```python
class CryptoEnvAlpaca:
    def __init__(self, config, env_params, ..., macro_array=None, use_macro=False):
        # ... existing code ...

        # Macro features
        self.macro_array = macro_array
        self.use_macro = use_macro and macro_array is not None

        if self.use_macro:
            self.state_dim += self.macro_array.shape[1]  # Add macro features to state

    def _get_state(self):
        # ... existing code to get price, tech, sentiment ...

        # Add macro features
        if self.use_macro:
            macro_features = self.macro_array[self.time]
            state = np.concatenate([state, macro_features])

        return state
```

### Expected Impact
- **+10-15% accuracy improvement** from macro context
- **Better regime detection** - QE vs QT environments
- **Early recession signals** - yield curve, stress indices
- **Rate sensitivity** - model learns Fed policy impact

---

## Enhancement 3: Fed-Focused Sentiment Analysis

### Rationale
Generic sentiment (Fear & Greed) is useful but noisy. Fed-specific sentiment provides:
- **Policy direction** - Hawkish vs dovish signals
- **Surprise factor** - Unexpected Fed moves
- **Forward guidance** - What the Fed plans to do
- **Research insights** - Fed papers on digital assets, regulation

### Data Sources

#### **Tier 1: Official Fed Communications**

| Source | Frequency | How to Access | Value |
|--------|-----------|---------------|-------|
| **FOMC Statements** | 8x/year | federalreserve.gov | Policy direction |
| **FOMC Minutes** | 8x/year (3 weeks after meeting) | federalreserve.gov | Detailed discussions |
| **Fed Chair Speeches** | Weekly | federalreserve.gov/newsevents | Real-time policy hints |
| **Fed Economic Projections** | Quarterly | federalreserve.gov/monetarypolicy | Dot plot, forecasts |

#### **Tier 2: Fed Research & Papers**

| Source | Frequency | How to Access | Value |
|--------|-----------|---------------|-------|
| **Fed Working Papers** | Weekly | federalreserve.gov/econres | Academic research |
| **Fed Notes (FEDS Notes)** | Weekly | federalreserve.gov/econresdata | Policy analysis |
| **Regional Fed Research** | Daily |各地Fed网站 | Economic conditions |
| **Fed Digital Currency Research** | Ad-hoc | Search "CBDC" on Fed sites | Crypto-specific |

#### **Tier 3: Fed-Watching Services**

| Source | Frequency | How to Access | Value |
|--------|-----------|---------------|-------|
| **Bloomberg Fed Articles** | Real-time | Bloomberg API (paid) | Market interpretation |
| **Fed Watch Tool (CME)** | Real-time | cmegroup.com | Rate hike probabilities |
| **Fed transcripts** | 5-year delay | federalreserve.gov | Historical context |

### Implementation Approaches

#### **Approach 1: Web Scraping Fed Publications**

```python
# sentiment_fed_scraper.py - NEW FILE

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import re

class FedSentimentScraper:
    """
    Scrapes Federal Reserve publications for sentiment analysis.
    """

    BASE_URLS = {
        'speeches': 'https://www.federalreserve.gov/newsevents/speeches.htm',
        'statements': 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm',
        'notes': 'https://www.federalreserve.gov/econres/notes/default.htm',
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Research Bot for Academic Purposes)'
        })

    def scrape_fomc_statements(self, start_date, end_date):
        """
        Scrape FOMC statements within date range.

        Returns:
            DataFrame with columns: date, title, text, url
        """
        statements = []

        # Scrape calendar page
        response = self.session.get(self.BASE_URLS['statements'])
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find statement links (implementation depends on HTML structure)
        # This is pseudocode - actual implementation needs to parse real HTML
        for statement_link in soup.find_all('a', href=re.compile(r'/monetarypolicy/files/.*\.pdf')):
            date_str = statement_link.get('data-date')  # Example
            if not date_str:
                continue

            date = datetime.strptime(date_str, '%Y-%m-%d')
            if start_date <= date <= end_date:
                # Download PDF and extract text
                pdf_url = 'https://www.federalreserve.gov' + statement_link['href']
                text = self._extract_text_from_pdf(pdf_url)

                statements.append({
                    'date': date,
                    'type': 'FOMC Statement',
                    'text': text,
                    'url': pdf_url
                })

        return pd.DataFrame(statements)

    def scrape_fed_speeches(self, start_date, end_date, speaker=None):
        """
        Scrape Fed official speeches.

        Args:
            start_date: Start date
            end_date: End date
            speaker: Filter by speaker (e.g., 'Powell', 'Yellen')

        Returns:
            DataFrame with speeches
        """
        speeches = []

        response = self.session.get(self.BASE_URLS['speeches'])
        soup = BeautifulSoup(response.content, 'html.parser')

        # Parse speech list (pseudocode)
        for speech_div in soup.find_all('div', class_='event-item'):
            date_elem = speech_div.find('time')
            speaker_elem = speech_div.find('span', class_='speaker')
            link_elem = speech_div.find('a', href=True)

            if not all([date_elem, speaker_elem, link_elem]):
                continue

            date = datetime.strptime(date_elem.get('datetime'), '%Y-%m-%d')
            speaker_name = speaker_elem.text.strip()

            if start_date <= date <= end_date:
                if speaker is None or speaker.lower() in speaker_name.lower():
                    # Get full speech text
                    speech_url = 'https://www.federalreserve.gov' + link_elem['href']
                    text = self._scrape_speech_text(speech_url)

                    speeches.append({
                        'date': date,
                        'speaker': speaker_name,
                        'title': link_elem.text.strip(),
                        'text': text,
                        'url': speech_url
                    })

        return pd.DataFrame(speeches)

    def scrape_fed_notes(self, start_date, end_date, topic_filter=None):
        """
        Scrape FEDS Notes (policy analysis by Fed staff).

        Args:
            start_date: Start date
            end_date: End date
            topic_filter: Keywords to filter (e.g., 'crypto', 'digital', 'inflation')

        Returns:
            DataFrame with Fed Notes
        """
        # Similar implementation to above
        pass

    def _extract_text_from_pdf(self, pdf_url):
        """Extract text from PDF using pdfplumber or PyPDF2."""
        import pdfplumber
        import io

        response = self.session.get(pdf_url)
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            text = '\n'.join(page.extract_text() for page in pdf.pages)
        return text

    def _scrape_speech_text(self, speech_url):
        """Scrape speech text from HTML page."""
        response = self.session.get(speech_url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Fed speeches usually in a specific div (check actual HTML)
        speech_div = soup.find('div', id='article')
        if speech_div:
            return speech_div.get_text(strip=True)
        return ""
```

#### **Approach 2: Sentiment Analysis on Fed Text**

```python
# sentiment_fed_analyzer.py - NEW FILE

import ollama
import pandas as pd
import numpy as np
from datetime import datetime

class FedSentimentAnalyzer:
    """
    Analyzes Fed communications for monetary policy sentiment.
    """

    def __init__(self, model='mvkvl/sentiments:aya'):
        """
        Initialize with Ollama sentiment model.

        Args:
            model: Ollama model for sentiment analysis
        """
        self.model = model

    def analyze_fomc_statement(self, statement_text):
        """
        Analyze FOMC statement for hawkish/dovish sentiment.

        Returns:
            dict with sentiment scores
        """
        prompt = f"""
Analyze this FOMC statement and rate the monetary policy stance on a scale:

-1.0 = Very Dovish (easy money, rate cuts likely)
 0.0 = Neutral
+1.0 = Very Hawkish (tight money, rate hikes likely)

Consider:
- Language about inflation (persistent = hawkish, subsiding = dovish)
- Rate guidance (higher for longer = hawkish, patient = dovish)
- Balance sheet policy (QT = hawkish, QE = dovish)
- Economic outlook (strong = hawkish, weak = dovish)

Statement:
{statement_text[:2000]}  # Limit to 2000 chars for model context

Respond with ONLY a number between -1.0 and 1.0, no explanation.
"""

        response = ollama.generate(model=self.model, prompt=prompt)

        try:
            sentiment_score = float(response['response'].strip())
            # Clamp to [-1, 1]
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
        except:
            sentiment_score = 0.0  # Neutral if parsing fails

        # Also extract key phrases
        keywords = self._extract_policy_keywords(statement_text)

        return {
            'sentiment_score': sentiment_score,
            'hawkish_keywords': keywords['hawkish'],
            'dovish_keywords': keywords['dovish'],
            'confidence': self._calculate_confidence(keywords)
        }

    def analyze_speech(self, speech_text):
        """
        Analyze Fed speech for policy signals.

        Returns:
            dict with sentiment and key topics
        """
        # Similar to above, but also extract:
        # - Topics discussed (inflation, employment, growth)
        # - Urgency level
        # - Surprise factor (how different from market expectations)

        prompt = f"""
Analyze this Federal Reserve speech. Extract:

1. Policy Stance (-1 to +1): Dovish to Hawkish
2. Inflation Concern (0 to 1): Low to High
3. Economic Optimism (0 to 1): Pessimistic to Optimistic
4. Urgency (0 to 1): Patient to Urgent

Speech excerpt:
{speech_text[:2000]}

Respond in format:
STANCE: [number]
INFLATION: [number]
OPTIMISM: [number]
URGENCY: [number]
"""

        response = ollama.generate(model=self.model, prompt=prompt)

        # Parse response
        results = self._parse_multi_dimension_response(response['response'])

        return results

    def analyze_fed_research(self, paper_abstract, paper_title):
        """
        Analyze Fed research paper for crypto/digital asset relevance.

        Returns:
            dict with relevance score and implications
        """
        prompt = f"""
This is a Federal Reserve research paper. Assess its relevance to cryptocurrency markets.

Title: {paper_title}

Abstract:
{paper_abstract}

Rate relevance (0 to 1):
0.0 = Not relevant to crypto
0.5 = Indirectly relevant (macro factors)
1.0 = Directly about crypto/digital assets

Also classify sentiment toward crypto:
-1.0 = Very negative (regulatory concerns)
 0.0 = Neutral/academic
+1.0 = Very positive (innovation, benefits)

Respond in format:
RELEVANCE: [number]
CRYPTO_SENTIMENT: [number]
"""

        response = ollama.generate(model=self.model, prompt=prompt)
        results = self._parse_research_response(response['response'])

        return results

    def _extract_policy_keywords(self, text):
        """Extract hawkish/dovish keywords from text."""
        text_lower = text.lower()

        hawkish_keywords = [
            'persistent inflation', 'tight', 'restrictive', 'elevated',
            'vigilant', 'determined', 'higher for longer', 'premature',
            'normalize', 'reduce balance sheet'
        ]

        dovish_keywords = [
            'patient', 'accommodative', 'supportive', 'gradual',
            'measured', 'subsiding', 'moderating', 'appropriate',
            'data-dependent', 'cautious'
        ]

        found_hawkish = [kw for kw in hawkish_keywords if kw in text_lower]
        found_dovish = [kw for kw in dovish_keywords if kw in text_lower]

        return {
            'hawkish': found_hawkish,
            'dovish': found_dovish
        }

    def _calculate_confidence(self, keywords):
        """
        Calculate confidence in sentiment based on keyword strength.

        More keywords = higher confidence
        """
        total_keywords = len(keywords['hawkish']) + len(keywords['dovish'])
        return min(total_keywords / 5.0, 1.0)  # Cap at 1.0

    def _parse_multi_dimension_response(self, response_text):
        """Parse multi-line response from LLM."""
        results = {}

        for line in response_text.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                try:
                    results[key.strip().lower()] = float(value.strip())
                except:
                    results[key.strip().lower()] = 0.0

        return results

    def _parse_research_response(self, response_text):
        """Parse research paper analysis response."""
        # Similar to above
        return self._parse_multi_dimension_response(response_text)
```

#### **Approach 3: CME FedWatch Tool Integration**

```python
# sentiment_fed_watch.py - NEW FILE

import requests
import pandas as pd
from datetime import datetime

class FedWatchProcessor:
    """
    Processes CME FedWatch data for rate hike probabilities.

    FedWatch shows market-implied probabilities of Fed rate decisions.
    """

    def __init__(self):
        # CME FedWatch data is publicly available
        self.base_url = 'https://www.cmegroup.com/CmeWS/mvc/Quotes/Future'

    def get_rate_probabilities(self, target_date=None):
        """
        Get probability distribution of Fed Funds Rate for target date.

        Returns:
            DataFrame with rate levels and probabilities
        """
        # Note: This is pseudocode - actual implementation depends on CME API
        # May need to scrape webpage or use unofficial API

        # Example data structure:
        probabilities = {
            '4.50-4.75%': 0.15,  # 15% chance of this rate
            '4.75-5.00%': 0.60,  # 60% chance
            '5.00-5.25%': 0.25,  # 25% chance
        }

        return pd.DataFrame(probabilities.items(), columns=['rate_range', 'probability'])

    def compute_expected_rate(self, probabilities_df):
        """
        Compute probability-weighted expected rate.

        Returns:
            float: Expected Fed Funds Rate
        """
        # Parse rate ranges and compute weighted average
        # Example: 4.50-4.75% → midpoint = 4.625%

        expected_rate = 0.0
        for _, row in probabilities_df.iterrows():
            rate_range = row['rate_range']
            prob = row['probability']

            # Extract midpoint
            low, high = self._parse_rate_range(rate_range)
            midpoint = (low + high) / 2

            expected_rate += midpoint * prob

        return expected_rate

    def compute_surprise_index(self, current_expected, previous_expected):
        """
        Compute how much Fed expectations changed (surprise factor).

        Large changes = market repricing = volatility

        Returns:
            float: Change in expected rate (basis points)
        """
        return (current_expected - previous_expected) * 100  # In basis points

    def _parse_rate_range(self, rate_range_str):
        """Parse '4.50-4.75%' → (4.50, 4.75)"""
        import re
        numbers = re.findall(r'\d+\.\d+', rate_range_str)
        return float(numbers[0]), float(numbers[1])
```

### Fed Sentiment Features

For each timepoint, compute:

1. **Policy Stance**:
   - `fed_sentiment_score`: -1 (dovish) to +1 (hawkish)
   - `fed_sentiment_change_7d`: How much stance changed in past week
   - `fed_sentiment_volatility`: Stability of messaging

2. **Rate Expectations**:
   - `expected_fed_funds_rate`: Market-implied future rate
   - `rate_surprise_index`: Unexpected changes in expectations
   - `hike_probability_next_meeting`: % chance of rate hike

3. **Communication Metrics**:
   - `fed_speech_frequency`: # of speeches per week (high = important period)
   - `fomc_days_until_next`: Days until next FOMC meeting
   - `fomc_days_since_last`: Days since last meeting

4. **Research Signals**:
   - `fed_crypto_research_sentiment`: Fed's view on crypto (from papers)
   - `fed_cbdc_mentions`: Mentions of digital currency

### Usage Integration

```python
# In data loading script

from sentiment_fed_scraper import FedSentimentScraper
from sentiment_fed_analyzer import FedSentimentAnalyzer
from sentiment_fed_watch import FedWatchProcessor

# Scrape Fed communications
scraper = FedSentimentScraper()
statements = scraper.scrape_fomc_statements(start_date, end_date)
speeches = scraper.scrape_fed_speeches(start_date, end_date)

# Analyze sentiment
analyzer = FedSentimentAnalyzer(model='mvkvl/sentiments:aya')

fed_sentiment_series = []
for _, statement in statements.iterrows():
    sentiment = analyzer.analyze_fomc_statement(statement['text'])
    fed_sentiment_series.append({
        'date': statement['date'],
        'sentiment_score': sentiment['sentiment_score'],
        'confidence': sentiment['confidence']
    })

fed_sentiment_df = pd.DataFrame(fed_sentiment_series)

# Get rate probabilities
fedwatch = FedWatchProcessor()
rate_probs = fedwatch.get_rate_probabilities()
expected_rate = fedwatch.compute_expected_rate(rate_probs)

# Combine into features
fed_features = pd.DataFrame({
    'fed_sentiment': fed_sentiment_df['sentiment_score'],
    'expected_rate': expected_rate,
    'surprise_index': fedwatch.compute_surprise_index(...),
})

# Resample to crypto timeframe (forward-fill)
fed_features_resampled = fed_features.resample('1h').ffill()

# Save
import pickle
with open('data/1h_1440/fed_sentiment_array', 'wb') as f:
    pickle.dump(fed_features_resampled.values, f)
```

### Expected Impact
- **+8-12% accuracy improvement** from policy-aware features
- **Better event prediction** - FOMC meetings, Fed speeches
- **Regime switching** - QE to QT transitions
- **Surprise detection** - Hawkish surprise = sell signal

---

## Implementation Roadmap

### Phase 1: Multi-Asset Correlations (2-3 weeks)

**Week 1: Data Infrastructure**
- [x] Create `processor_MultiAsset.py`
- [ ] Download Tier 1 assets (SPY, QQQ, DXY, GLD, VIX)
- [ ] Implement resampling to match crypto timeframes
- [ ] Validate data quality and alignment

**Week 2: Feature Engineering**
- [ ] Compute rolling correlations (7d, 30d windows)
- [ ] Implement divergence signals
- [ ] Create relative strength ratios
- [ ] Normalize features for RL environment

**Week 3: Integration & Testing**
- [ ] Update environment to accept multi-asset features
- [ ] Modify state dimension calculation
- [ ] Run baseline tests (accuracy before/after)
- [ ] Hyperparameter tuning with new features

**Expected Outcome**: +5-10% improvement in accuracy

---

### Phase 2: FRED Economic Indicators (3-4 weeks)

**Week 1: FRED API Setup**
- [ ] Get FRED API key
- [ ] Install `fredapi` package
- [ ] Create `processor_FRED.py`
- [ ] Download Tier 1 series (DFF, WALCL, EFFR, RRPONTSYD)

**Week 2: Feature Engineering**
- [ ] Resample FRED data to crypto timeframe
- [ ] Compute rate of change features
- [ ] Create Z-scores for each series
- [ ] Build composite features (real rate, liquidity proxy)

**Week 3: Inflation & Stress Indicators**
- [ ] Download Tier 2 series (CPI, PCE, UNRATE)
- [ ] Download Tier 3 series (T10Y2Y, VIX, Dollar Index)
- [ ] Create recession prediction features
- [ ] Build Fed balance sheet expansion features

**Week 4: Integration & Validation**
- [ ] Update environment with macro features
- [ ] Test macro-aware training runs
- [ ] Analyze feature importance (which FRED series matter most?)
- [ ] Publish results

**Expected Outcome**: +10-15% improvement (cumulative: +15-25%)

---

### Phase 3: Fed-Focused Sentiment (4-5 weeks)

**Week 1: Data Collection Pipeline**
- [ ] Create `sentiment_fed_scraper.py`
- [ ] Implement FOMC statement scraper
- [ ] Implement Fed speech scraper
- [ ] Test PDF extraction for Fed documents

**Week 2: Sentiment Analysis**
- [ ] Create `sentiment_fed_analyzer.py`
- [ ] Fine-tune prompts for hawkish/dovish detection
- [ ] Implement multi-dimensional analysis (inflation, urgency, optimism)
- [ ] Validate against known hawkish/dovish statements

**Week 3: Research & FedWatch**
- [ ] Scrape Fed research papers on crypto/CBDC
- [ ] Integrate CME FedWatch data (rate probabilities)
- [ ] Build surprise index (unexpected Fed shifts)
- [ ] Create calendar features (days to/from FOMC)

**Week 4: Feature Engineering**
- [ ] Compute Fed sentiment time series
- [ ] Resample to crypto timeframe
- [ ] Create sentiment change features
- [ ] Build event impact features

**Week 5: Integration & Testing**
- [ ] Update environment with Fed sentiment
- [ ] A/B test: generic sentiment vs Fed sentiment
- [ ] Measure improvement on FOMC meeting days
- [ ] Publish final results

**Expected Outcome**: +8-12% improvement (cumulative: +23-37%)

---

## Total Expected Impact

| Enhancement | Accuracy Gain | Cumulative | Implementation Time |
|-------------|---------------|------------|---------------------|
| **Baseline (current)** | - | 0% | - |
| **Multi-Asset Correlations** | +5-10% | +5-10% | 2-3 weeks |
| **FRED Macro Indicators** | +10-15% | +15-25% | 3-4 weeks |
| **Fed-Focused Sentiment** | +8-12% | +23-37% | 4-5 weeks |

**Total: +23-37% improvement in predictive accuracy**

---

## Resource Requirements

### APIs & Data
- **FRED API**: Free (30 calls/minute limit)
- **Yahoo Finance**: Free via yfinance
- **CME FedWatch**: Free (scraping)
- **Fed website**: Free (scraping)
- **Ollama models**: Already available locally

### Compute
- **Training time**: +20-30% longer (more features)
- **Memory**: +30-40% (larger state space)
- **GPU**: Same (RTX 3070 sufficient)

### Development
- **Phase 1**: 1 developer, 2-3 weeks
- **Phase 2**: 1 developer, 3-4 weeks
- **Phase 3**: 1 developer, 4-5 weeks
- **Total**: ~9-12 weeks for full implementation

---

## Alternative: Quick Wins (1-2 weeks)

If you want faster results, prioritize:

1. **SPY & VIX correlation** (2 days)
   - Add S&P 500 and VIX to features
   - Immediate signal improvement

2. **DFF (Fed Funds Rate)** (1 day)
   - Single most important macro indicator
   - Free via FRED

3. **BTC Dominance** (1 day)
   - Already crypto-native
   - Strong altcoin predictor

**Expected gain**: +8-12% in 1 week

---

## Next Steps

1. **Review this roadmap** and confirm priorities
2. **Set up FRED API key** (5 minutes)
3. **Start with Phase 1** (Multi-Asset) for quick wins
4. **Parallel track**: Set up Fed scraper while Phase 1 runs

Let me know which phase you'd like to start with, and I can create the implementation code!
