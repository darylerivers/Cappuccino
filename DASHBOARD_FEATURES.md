# Dashboard Features - Enhanced Quick Start Menu

## 🎯 New Features Added to Option 5

The **"Show Trial Dashboard"** option now has **3 sub-options**:

```bash
./quick_start_automated_training.sh
# Select: 5 - Show Trial Dashboard
```

### Option 1: Live Dashboard (auto-refresh)
The original real-time dashboard that updates every 30 seconds.

**Shows**:
- Current study name
- Running workers (PID, CPU, MEM)
- Recent trials with VIN codes
- Top archived trials
- Paper trading status

**Usage**: Auto-refreshes, press Ctrl+C to exit

---

### Option 2: Show Current Statistics ⭐ NEW
Displays comprehensive statistics for the current training study.

**Shows**:
- Total completed trials
- **Average Sharpe ratio** across all trials
- Min/Max Sharpe ratio
- **Grade distribution** with visual bars

**Example Output**:
```
Study: cappuccino_tightened_20260201
======================================================================
Total Completed Trials: 341
Average Sharpe Ratio:   0.0922
Min Sharpe Ratio:       -0.0110
Max Sharpe Ratio:       0.1767

Grade Distribution:
  ✅ B:  52 ( 15.2%) ███████
  🔵 C: 150 ( 44.0%) █████████████████████
  ⚠️ D:  71 ( 20.8%) ██████████
  ❌ F:  68 ( 19.9%) █████████
```

**Use Case**: Quick overview of training performance without live monitoring

---

### Option 3: Show Top 10 Trials ⭐ NEW
Lists the **top 10 best-performing trials** from the current study.

**Shows**:
- Rank (1-10)
- Trial number
- Full VIN code (encoded hyperparameters)
- Grade icon and letter
- Sharpe ratio

**Example Output**:
```
Study: cappuccino_tightened_20260201
================================================================================
Rank   Trial    VIN Code                                           Grade   Sharpe
------ -------- -------------------------------------------------- ------- ----------
1      #1176    PPO-B-N1344B0-L1.1E4G97-LB1TD0-20260206           ✅ B       0.1767
2      #1170    PPO-B-N1408B0-L8.5E5G97-LB1TD0-20260206           ✅ B       0.1767
3      #982     PPO-B-N1088B0-L5.6E5G98-LB1TD0-20260206           ✅ B       0.1767
4      #1161    PPO-B-N1408B0-L1E6G98-LB1TD0-20260206             ✅ B       0.1753
5      #963     PPO-B-N1408B0-L7.2E5G98-LB1TD0-20260206           ✅ B       0.1745
...
```

**Use Case**:
- Identify best hyperparameter combinations
- See which trials are worth archiving
- Compare top performers at a glance

---

## 🎨 How to Use

### Quick Access
```bash
./quick_start_automated_training.sh
```

**Menu Flow**:
1. Select: `5` - Show Trial Dashboard
2. Select sub-option:
   - `1` - Live Dashboard
   - `2` - Show Statistics
   - `3` - Show Top 10
   - `0` - Back to main menu

### Direct Python Access

**Statistics**:
```bash
# (The menu runs this internally)
python -c "... statistics code ..."
```

**Top 10**:
```bash
# (The menu runs this internally)
python -c "... top 10 code ..."
```

---

## 📊 Understanding the Output

### Grade Distribution Visualization

The bar chart shows the percentage distribution:
```
  ✅ B:  52 ( 15.2%) ███████
```
- `✅ B`: Grade icon and letter
- `52`: Number of trials with this grade
- `15.2%`: Percentage of total trials
- `███████`: Visual bar (each █ = 2%)

### VIN Code in Top 10

Example VIN: `PPO-B-N1408B0-L7.2E5G98-LB1TD0-20260206`

Breakdown:
- `PPO`: Model type
- `B`: Grade (Good - Sharpe 0.15-0.20)
- `N1408`: Network dimension = 1408
- `B0`: Batch size (0 likely means default/auto)
- `L7.2E5`: Learning rate = 7.2×10⁻⁵
- `G98`: Gamma = 0.98
- `LB1`: Lookback = 1
- `TD0`: Time decay = 0.0
- `20260206`: Trained on Feb 6, 2026

---

## 🎯 Use Cases

### Monitor Training Progress
```bash
# Option 1: Live Dashboard
./quick_start_automated_training.sh → 5 → 1
```
- Real-time updates
- See trials as they complete
- Monitor worker status

### Check Overall Performance
```bash
# Option 2: Statistics
./quick_start_automated_training.sh → 5 → 2
```
- See if training is improving
- Check average Sharpe ratio
- Identify grade distribution

### Find Best Hyperparameters
```bash
# Option 3: Top 10
./quick_start_automated_training.sh → 5 → 3
```
- See which configurations work best
- Compare VIN codes to spot patterns
- Identify trials to archive manually

---

## 🔧 Technical Details

### Current Study Detection

The dashboard automatically detects the current study from:
1. `.current_study` file (if exists)
2. Latest study in database (fallback)

### Database Query

- Uses `databases/optuna_cappuccino.db`
- Joins `trials` and `trial_values` tables
- Filters by `state = 'COMPLETE'`
- Sorts by Sharpe ratio (descending)

### Grade Calculation

Grades are calculated on-the-fly:
- **S**: Sharpe ≥ 0.30 (Elite)
- **A**: Sharpe ≥ 0.20 (Excellent - Top 10%)
- **B**: Sharpe ≥ 0.15 (Good)
- **C**: Sharpe ≥ 0.10 (Fair)
- **D**: Sharpe ≥ 0.05 (Poor)
- **F**: Sharpe < 0.05 (Failed)

---

## 📝 Summary

The enhanced dashboard menu provides **3 complementary views** of your training:

1. **Live Dashboard** → Real-time monitoring
2. **Statistics** → Average performance & distribution
3. **Top 10** → Best trials with VIN codes

All accessible from the quick start menu under option 5!

**Quick Start**:
```bash
./quick_start_automated_training.sh
# Select: 5
# Then: 1, 2, or 3
```
