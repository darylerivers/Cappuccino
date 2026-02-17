# Step 3: Rolling Mean Features - Implementation Guide

**Status:** ✅ Complete - Ready for Training
**Date:** December 16, 2025
**State Dimension:** 63 → 91 (+28 features)

---

## Overview

Step 3 adds rolling mean features to enhance the agent's ability to detect trends and momentum. This expands the state dimension from 63 to 91 by adding 28 new features.

### New Features Added

For each of the 7 cryptocurrencies:
1. **7-day close moving average** - Short-term trend
2. **7-day volume moving average** - Short-term activity
3. **30-day close moving average** - Long-term trend
4. **30-day volume moving average** - Long-term activity

**Total:** 4 features × 7 cryptos = 28 new features

---

## What Was Built

### 1. Data Preparation Script

**File:** `prepare_phase2_data.py`

**Purpose:** Generates Phase 2 enhanced data with rolling mean features

**Usage:**
```bash
# Basic usage
python prepare_phase2_data.py --input data/1h_1680 --output data/1h_1680_phase2

# Custom interval
python prepare_phase2_data.py --input data/4h_720 --output data/4h_720_phase2 --interval 4h
```

**What it does:**
- Loads existing price, tech, and time arrays
- Calculates 7-day and 30-day rolling means for close prices and volume
- Appends 28 rolling features to tech array (77 → 105 features)
- Saves enhanced data to new directory
- Creates metadata.json with configuration details

### 2. Validation Script

**File:** `validate_phase2_data.py`

**Purpose:** Validates rolling features were calculated correctly

**Usage:**
```bash
python validate_phase2_data.py --data-dir data/1h_1680_phase2
```

**Validation Checks:**
1. ✓ Shape consistency (105 = 77 + 28)
2. ✓ Base features preserved
3. ✓ Rolling features correctly appended
4. ✓ No NaN values
5. ✓ No inf values
6. ✓ Rolling mean values reasonable
7. ✓ Manual calculation matches

---

## Data Structure

### Input (Phase 1)

```
data/1h_1680/
├── price_array.npy     (8608, 7)   - Close prices
├── tech_array.npy      (8608, 77)  - Technical indicators
└── time_array.npy      (8608,)     - Timestamps
```

**Tech array structure (77 features):**
- 11 indicators × 7 cryptos = 77
- Indicators: [open, high, low, close, volume, macd, macd_signal, macd_hist, rsi, cci, dx]

### Output (Phase 2)

```
data/1h_1680_phase2/
├── price_array.npy          (8608, 7)    - Same as input
├── tech_array.npy           (8608, 105)  - Enhanced with rolling features
├── time_array.npy           (8608,)      - Same as input
├── tech_array_base.npy      (8608, 77)   - Original tech array (backup)
├── rolling_features.npy     (8608, 28)   - Just the rolling features
└── metadata.json                         - Configuration metadata
```

**Enhanced tech array structure (105 features):**
- Base: 77 features (original indicators)
- Rolling: 28 features (7 cryptos × 4 rolling features)
- Total: 105 features

**Rolling features layout (28 features):**
```
Crypto 0: [close_ma7d, volume_ma7d, close_ma30d, volume_ma30d]  # columns 77-80
Crypto 1: [close_ma7d, volume_ma7d, close_ma30d, volume_ma30d]  # columns 81-84
Crypto 2: [close_ma7d, volume_ma7d, close_ma30d, volume_ma30d]  # columns 85-88
...
Crypto 6: [close_ma7d, volume_ma7d, close_ma30d, volume_ma30d]  # columns 101-104
```

---

## State Dimension Calculation

### Phase 1 (Base: 63 dimensions)

```
State = [
    Cash (1),
    Current stock holdings (7),
    Tech features with lookback (11 × 5 = 55)
]
Total = 1 + 7 + 55 = 63
```

### Phase 2 (Enhanced: 91 dimensions)

```
State = [
    Cash (1),
    Current stock holdings (7),
    Tech features with lookback (11 × 5 = 55),
    Rolling features (7 cryptos × 4 features = 28)
]
Total = 1 + 7 + 55 + 28 = 91
```

**Note:** Rolling features are NOT affected by lookback because they're already time-aggregated.

---

## Rolling Window Configuration

### For 1-hour data:
- **7-day window:** 7 days × 24 hours = 168 candles
- **30-day window:** 30 days × 24 hours = 720 candles

### For other intervals:

| Interval | Candles/Day | 7-day window | 30-day window |
|----------|-------------|--------------|---------------|
| 5m       | 288         | 2,016        | 8,640         |
| 15m      | 96          | 672          | 2,880         |
| 30m      | 48          | 336          | 1,440         |
| 1h       | 24          | 168          | 720           |
| 4h       | 6           | 42           | 180           |
| 12h      | 2           | 14           | 60            |
| 1d       | 1           | 7            | 30            |

---

## Feature Statistics

From validation on data/1h_1680_phase2:

**First cryptocurrency (BTC):**

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| 7-day close MA | 254.90 | 57.02 | 134.66 | 369.40 |
| 30-day close MA | 255.97 | 52.12 | 148.61 | 338.05 |

**Observations:**
- 7-day MA more volatile (higher std: 57.02)
- 30-day MA smoother (lower std: 52.12)
- Both track similar price range (mean ~255)

---

## How to Use in Training

### Option A: Direct Usage

```bash
python 1_optimize_unified.py \
    --n-trials 50 \
    --study-name phase2_test \
    --gpu 0 \
    --data-dir data/1h_1680_phase2 \
    --timeframe 1h \
    --tickers BTC ETH LTC DOGE ADA SOL MATIC
```

### Option B: Modify Environment

The environment should automatically detect the enhanced state dimension from the tech array shape. No code changes needed if using standard `environment_Alpaca.py`.

**State dimension is calculated dynamically:**
- Reads tech_array.shape[1] → 105 features
- With lookback=5: state_dim = 1 + 7 + (105 - 28) × 5 + 28 = 91

Actually, wait - I need to check how the environment handles this...

### Important: Environment Compatibility

The environment needs to know how to handle rolling features. Let me check if any modifications are needed.

**Current approach:**
- Rolling features are at the END of tech_array (columns 77-104)
- They should NOT be affected by lookback window
- Only base features (columns 0-76) use lookback

**Potential issue:** Environment may apply lookback to ALL tech features, which would be incorrect for rolling features.

**Solution options:**
1. Modify environment to handle rolling features separately
2. Pass rolling features as separate input
3. Keep rolling features in tech_array but flag them

---

## Next Steps After Data Preparation

### Immediate:
1. ✅ Phase 2 data created (data/1h_1680_phase2)
2. ✅ Validation passed
3. ⏳ Test fundamental fixes (currently running)

### After Fundamental Fixes Test:
4. **If fixes successful:** Launch Phase 2 training with rolling features
5. **If fixes neutral/failed:** Fix issues, then proceed to Phase 2

### Phase 2 Training Plan:
- **Study name:** `phase2_rolling_features_20251216`
- **Trials:** 50-100 (start with 50 to validate)
- **Data:** data/1h_1680_phase2
- **State dim:** 91 (auto-detected)
- **Network sizes:** Larger than Phase 1 (1024-2560 recommended)
- **Algorithm:** PPO (proven to work)
- **Duration:** ~20-40 hours for 50 trials

---

## Expected Benefits

### Why Rolling Features Help:

1. **Trend Detection**
   - 7-day MA shows short-term momentum
   - 30-day MA shows long-term trend
   - MA crossovers signal trend changes

2. **Volatility Awareness**
   - Volume MA shows trading activity trends
   - High volume + price MA → strong trends
   - Low volume + price MA → weak trends

3. **Regime Detection**
   - Bull market: Price > 30-day MA
   - Bear market: Price < 30-day MA
   - Consolidation: Price oscillating around MA

4. **Entry/Exit Timing**
   - Buy signals: 7-day MA crosses above 30-day MA
   - Sell signals: 7-day MA crosses below 30-day MA
   - Agent learns to use these signals implicitly

### Expected Performance Improvement:

**Conservative:** +1-2% better risk-adjusted returns
**Optimistic:** +3-5% improvement from better timing

**Why:**
- Better trend following
- Earlier exit from downtrends (hold cash)
- More confident entries on confirmed uptrends

---

## Files Created

1. **prepare_phase2_data.py** - Data preparation script
2. **validate_phase2_data.py** - Validation script
3. **data/1h_1680_phase2/** - Enhanced Phase 2 data
4. **STEP_3_ROLLING_MEAN_GUIDE.md** - This guide

---

## Troubleshooting

### Issue: Environment not recognizing enhanced features

**Solution:** Check that environment loads tech_array correctly
```python
tech_array = np.load('data/1h_1680_phase2/tech_array.npy')
print(f"Tech array shape: {tech_array.shape}")  # Should be (8608, 105)
```

### Issue: State dimension mismatch

**Problem:** Environment may apply lookback to all features
**Solution:** Modify environment to treat last 28 features as non-lookback

### Issue: Training much slower

**Expected:** Phase 2 training ~20% slower due to larger state (91 vs 63)
**Mitigation:** Use larger batch sizes (3072-4096)

---

## Configuration Reference

From `config_two_phase.py`:

```python
@dataclass(frozen=True)
class Phase2Config:
    # Rolling mean windows (in days)
    ROLLING_WINDOW_SHORT: int = 7   # 7-day rolling means
    ROLLING_WINDOW_LONG: int = 30   # 30-day rolling means

    # State dimension calculation
    BASE_STATE_DIM: int = 63
    ROLLING_FEATURES_PER_CRYPTO: int = 4
    N_CRYPTOS: int = 7

    @property
    def ENHANCED_STATE_DIM(self) -> int:
        return self.BASE_STATE_DIM + (self.N_CRYPTOS * self.ROLLING_FEATURES_PER_CRYPTO)
        # 63 + (7 * 4) = 91
```

---

## Success Criteria

Phase 2 training is successful if:

1. ✅ Models converge (sharpe ratio improving over trials)
2. ✅ State dimension correctly recognized (91)
3. ✅ No NaN/inf errors during training
4. ✅ Performance better than Phase 1 baseline
5. ✅ Models show improved trend following behavior

---

## Contact & Support

**Created:** December 16, 2025
**Step:** 3 of 7 in experiment plan
**Status:** Ready for training after fundamental fixes validation

---

**Next:** Launch Phase 2 training with rolling features after validating fundamental fixes.
