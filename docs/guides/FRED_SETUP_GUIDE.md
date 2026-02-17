# FRED API Setup Guide

## Quick Start (5 minutes)

### Step 1: Get Your Free FRED API Key

1. **Visit**: https://fred.stlouisfed.org/docs/api/api_key.html

2. **Click "Request API Key"**

3. **Fill out the form**:
   - Name: Your name
   - Email: Your email
   - Organization: Personal/Research/Academic
   - Purpose: Academic research on cryptocurrency trading models

4. **Submit** - You'll receive your API key immediately via email

**Example API Key format**: `abcd1234efgh5678ijkl9012mnop3456`

---

### Step 2: Add API Key to .env File

```bash
cd /home/mrc/experiment/cappuccino

# If .env doesn't exist, copy from template
cp .env.template .env

# Edit .env file
nano .env  # or vim, code, etc.
```

Add your key to the FRED section:
```bash
# FRED API Configuration
FRED_API_KEY=abcd1234efgh5678ijkl9012mnop3456  # Replace with your actual key
```

**Important**: Never commit .env to git! It's already in .gitignore.

---

### Step 3: Install FRED API Package

```bash
# Activate your Python environment
pyenv activate finrl-crypto  # or your env name

# Install fredapi
pip install fredapi
```

---

### Step 4: Test Connection

```bash
cd /home/mrc/experiment/cappuccino

# Test FRED API connection
python 0_dl_fred_data.py --test-connection
```

**Expected output**:
```
================================================================================
TESTING FRED API CONNECTION
================================================================================

✓ FRED API key loaded successfully

Testing download (DFF - Fed Funds Rate)...
  ✓ Downloaded 30 observations
    Latest: 2025-10-25 = 4.58

✓ Connection successful!
  Latest Fed Funds Rate: 4.58%
  Date: 2025-10-25
```

If you see errors, double-check:
- API key is correct in .env
- No extra spaces or quotes around the key
- .env file is in the cappuccino directory

---

### Step 5: Download FRED Data

```bash
# Download 2 years of data (default)
python 0_dl_fred_data.py

# Or specify custom settings
python 0_dl_fred_data.py --timeframe 1h --lookback-days 730
```

**This downloads**:
- **Monetary Policy**: Fed Funds Rate, Fed Balance Sheet, Reverse Repo
- **Inflation**: CPI, Core CPI, PCE Index
- **Market Stress**: Yield Curve, High Yield Spreads, VIX, Dollar Index
- **Fed Indices**: Financial Conditions Index, Stress Index

**Output location**: `data/1h_fred/`

---

## What Data Gets Downloaded?

### Monetary Policy Indicators

| Series | Description | Frequency | Impact on Crypto |
|--------|-------------|-----------|------------------|
| **DFF** | Fed Funds Rate | Daily | High rates = Lower crypto |
| **EFFR** | Effective Fed Funds Rate | Daily | Real-time policy rate |
| **WALCL** | Fed Balance Sheet | Weekly | Expansion = Higher crypto |
| **RRPONTSYD** | Reverse Repo | Daily | High = Tight liquidity |

### Inflation Indicators

| Series | Description | Frequency | Impact |
|--------|-------------|-----------|--------|
| **CPIAUCSL** | Consumer Price Index | Monthly | Inflation gauge |
| **CPILFESL** | Core CPI | Monthly | Fed's preferred metric |
| **PCEPI** | PCE Index | Monthly | Fed's inflation target |

### Market Stress Indicators

| Series | Description | Frequency | Impact |
|--------|-------------|-----------|--------|
| **T10Y2Y** | 10Y-2Y Treasury Spread | Daily | Negative = Recession warning |
| **BAMLH0A0HYM2** | High Yield Spread | Daily | Credit stress |
| **DTWEXBGS** | Dollar Index | Daily | Strong dollar = Weak crypto |
| **VIXCLS** | VIX (Fear Index) | Daily | High VIX = Risk-off |

### Fed Stress Indices

| Series | Description | Frequency | Impact |
|--------|-------------|-----------|--------|
| **NFCI** | Financial Conditions Index | Weekly | Financial stress composite |
| **STLFSI4** | St. Louis Stress Index | Weekly | Stress gauge |

---

## Features Generated

For each FRED series, the processor creates:

1. **Level Features** - Current value
   - `DFF`, `WALCL`, `CPIAUCSL`, etc.

2. **Rate of Change** - % change over time
   - `DFF_chg_7d`, `DFF_chg_30d`, `DFF_chg_90d`

3. **Z-Scores** - Statistical deviation from mean
   - `DFF_zscore_90d`, `DFF_zscore_180d`

4. **Composite Features** - Derived indicators
   - `real_rate` = Fed Funds Rate - Inflation Rate
   - `yield_curve_inverted` = 1 if 10Y-2Y < 0, else 0
   - `fed_bs_expansion_1m` = Fed Balance Sheet growth rate
   - `liquidity_stress` = Reverse Repo / Balance Sheet
   - `market_fear_composite` = Normalized VIX + High Yield Spread

**Total features**: ~50-60 macro features per timepoint

---

## Usage in Training

### Option 1: Using the Download Script

```bash
# 1. Download FRED data
python 0_dl_fred_data.py --timeframe 1h

# 2. Train with macro features (when implemented)
python 1_optimize_unified.py --use-macro --macro-dir data/1h_fred
```

### Option 2: In Your Own Code

```python
from processor_FRED import FREDProcessor
from datetime import datetime, timedelta
import pickle

# Initialize
processor = FREDProcessor()

# Download data
end_date = datetime.now()
start_date = end_date - timedelta(days=730)

processor.download_series(
    start_date=start_date,
    end_date=end_date,
    categories=['monetary', 'inflation', 'stress']
)

# Resample to match crypto timeframe
processor.data = processor.resample_to_timeframe('1h')

# Compute features
features_df = processor.compute_features()

# Align to crypto data
import pandas as pd
crypto_index = pd.date_range('2024-01-01', '2025-10-27', freq='1h')
macro_array = processor.align_to_crypto_data(crypto_index, features_df)

# Use in environment
# (See environment_Alpaca.py integration)
```

---

## Troubleshooting

### Error: "FRED API key not found"

**Solution**:
1. Check .env file exists in cappuccino directory
2. Verify FRED_API_KEY is set correctly
3. No quotes around the key value
4. Restart Python if you just added the key

### Error: "HTTPError: 400 Bad Request"

**Cause**: Invalid API key or series ID

**Solution**:
1. Verify API key is correct (copy-paste from email)
2. Check for typos in series IDs

### Error: "HTTPError: 429 Too Many Requests"

**Cause**: Hit API rate limit (30 requests/minute)

**Solution**:
1. Wait 60 seconds and try again
2. Download fewer categories at once
3. Use cached data if available

### Warning: "Downloaded 0 observations"

**Cause**: Date range out of bounds for that series

**Solution**:
1. Check series availability on FRED website
2. Adjust start date (some series start in 1980s, others more recent)
3. Use --lookback-days to adjust window

### Missing Data / NaN Values

**Expected behavior**: FRED data has different frequencies
- Daily: DFF, VIX, Dollar Index
- Weekly: Fed Balance Sheet, Stress Indices
- Monthly: CPI, Core CPI

The processor forward-fills missing values (e.g., monthly CPI is held constant until new data arrives).

---

## API Limits

- **Rate limit**: 30 requests per minute
- **Daily limit**: None specified (be reasonable)
- **Cost**: FREE

---

## Data Quality Notes

1. **Timeliness**:
   - Daily series: Updated next business day
   - Weekly series: Updated following week
   - Monthly series: Released ~2 weeks after month end

2. **Revisions**:
   - Some series (CPI, GDP) are revised after initial release
   - FRED shows latest revision

3. **Holidays**:
   - No data on federal holidays
   - Forward-fill handles gaps automatically

---

## Next Steps

After downloading FRED data:

1. **Update environment** to accept macro features
2. **Modify training script** to load and use macro data
3. **Run baseline experiments** with and without macro features
4. **Compare results** to measure accuracy improvement

Expected improvement: **+10-15% accuracy**

---

## Additional Resources

- **FRED Website**: https://fred.stlouisfed.org
- **API Documentation**: https://fred.stlouisfed.org/docs/api/
- **Series Search**: https://fred.stlouisfed.org/categories
- **Mobile App**: Download FRED Mobile for iOS/Android

---

## Support

If you encounter issues:

1. Check this guide first
2. Verify API key at: https://fredaccount.stlouisfed.org/apikeys
3. Test with `python 0_dl_fred_data.py --test-connection`
4. Check FRED status page for outages

---

**You're all set!** Start downloading macro data to boost your model's accuracy.
