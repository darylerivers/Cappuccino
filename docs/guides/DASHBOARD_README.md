# Paper Trading Dashboard - Setup Complete ✅

**Created:** January 17, 2026, 00:42 UTC
**Status:** Ready to use

---

## 🎯 What You Got

A real-time dashboard that compares your two paper traders side-by-side:
- **Ensemble Trader** (Top-20 model voting) - Currently running ✓
- **Single Model Trader** (Trial #861) - Has data, not currently running

---

## 🚀 Quick Start

### Option 1: Simple Launch
```bash
cd /opt/user-data/experiment/cappuccino
./start_dashboard.sh
```

### Option 2: Direct Launch
```bash
python paper_trading_dashboard.py
```

### Option 3: Custom Settings
```bash
# Faster refresh (every 30 seconds)
python paper_trading_dashboard.py --refresh 30

# Custom CSV files
python paper_trading_dashboard.py \
    --ensemble-csv paper_trades/my_ensemble.csv \
    --single-csv paper_trades/my_single.csv
```

---

## 📊 What It Shows

### Left Column: Ensemble Trader
- Status (Active/Idle/Stale)
- Total Return (% gain/loss)
- Sharpe Ratio
- Total Asset Value & Cash
- Trade counts (24h, 7d, all-time)
- Active positions with concentration %
- **Alerts if concentration >30%**

### Right Column: Single Model Trader
- Same metrics for easy comparison
- See which trader performs better
- Compare trade frequency
- Compare risk management (concentration)

### Updates Every 60 Seconds
- Auto-refreshes data
- No need to manually reload
- Press Ctrl+C to exit

---

## 📁 Files Created

```
✅ paper_trading_dashboard.py     # Main dashboard (540 lines)
✅ start_dashboard.sh              # Quick launcher
✅ check_traders_status.sh         # Status checker
✅ DASHBOARD_USAGE.md              # Detailed usage guide
✅ DASHBOARD_README.md             # This file
```

---

## ✨ Features

### 1. Real-Time Monitoring
- Live updates every 60 seconds
- See positions change as trades happen
- Monitor both traders simultaneously

### 2. Risk Alerts
- **Green:** Concentration <25% ✓ Safe
- **Yellow:** Concentration 25-30% ⚠️ Approaching limit
- **Red:** Concentration >30% 🚨 OVER LIMIT!

### 3. Performance Comparison
- Which trader has better returns?
- Which has better Sharpe ratio?
- Which trades more efficiently?

### 4. Status Indicators
- ✓ Active: Updated within 2 hours
- ⚠️ Idle: No update 2-24 hours
- ✗ Stale: No update >24 hours

---

## 🔍 Before You Start

### Check Everything Is Running
```bash
./check_traders_status.sh
```

**Expected Output:**
```
==========================================
Paper Trading Status Check
==========================================

1. Ensemble Trader Process:
   ✓ Running (PID: 1052065)

2. Single Model Trader Process:
   ⚠️  Not currently running (has data from earlier)

3. Ensemble Data File:
   ✓ Found: paper_trades/watchdog_session_20260116_183945.csv
   Lines: 4 | Size: 1.2K

4. Single Model Data File:
   ✓ Found: paper_trades/single_model_trial861.csv
   Lines: 8 | Size: 2.2K
```

---

## 💡 Usage Tips

### 1. Run in Background
Keep dashboard running in tmux:
```bash
tmux new -s dashboard
./start_dashboard.sh
# Detach: Ctrl+B then D
# Reattach later: tmux attach -t dashboard
```

### 2. Enhanced Display (Optional)
Install `rich` for prettier output:
```bash
pip install rich
```

This adds:
- Colored tables and panels
- Better visual hierarchy
- Status color coding (green/yellow/red)

**Dashboard works fine without it!**

### 3. Check Regularly
- **Morning:** See overnight trading activity
- **Evening:** Review daily performance
- **Every 24h:** Compare traders

---

## 📈 What to Look For

### Good Signs ✅
- Concentration 15-25% per asset
- Sharpe ratio >0.8
- 1-2 trades per day per asset
- Steady positive returns

### Warning Signs ⚠️
- Concentration >25% (approaching limit)
- Trade frequency >3/day per asset (overtrading)
- Sharpe ratio <0.5 (poor risk-adjusted returns)

### Action Required 🚨
- Concentration >30% (risk limit violated!)
- Trader status "Stale" >24h (process crashed)
- Returns <-5% (strategy failing)

---

## 🔧 Current Status

### Ensemble Trader
- ✅ Running (PID: 1052065)
- ✅ Writing to: `paper_trades/watchdog_session_20260116_183945.csv`
- ✅ 4 data points collected
- ✅ Last update: Jan 16, 21:02

### Single Model Trader
- ⚠️ Not currently running
- ✅ Has historical data: `paper_trades/single_model_trial861.csv`
- ✅ 8 data points collected
- ⚠️ Last update: Jan 16, 18:02 (stale)

**Note:** Dashboard will show ensemble trader in real-time, and single model data from its last run. You can restart the single model trader anytime to get fresh comparison data.

---

## 🎯 Next Steps

1. **Launch the dashboard:**
   ```bash
   ./start_dashboard.sh
   ```

2. **Let it run for 24-48 hours** to collect meaningful data

3. **Compare performance:**
   - Which trader has better returns?
   - Which manages risk better?
   - Which trades more efficiently?

4. **(Optional) Restart single model trader** if you want live comparison:
   ```bash
   # Check how it was launched before
   cat logs/single_model_trial861.log | head -20

   # Relaunch with same parameters
   ```

---

## 📚 Documentation

- **DASHBOARD_USAGE.md** - Complete usage guide with examples
- **DASHBOARD_README.md** - This file (quick start)
- Built-in help: `python paper_trading_dashboard.py --help`

---

## 🐛 Troubleshooting

### "No data showing"
```bash
# Check CSV files have data
wc -l paper_trades/*.csv

# Check traders are running
ps aux | grep paper_trader
```

### "Dashboard won't start"
```bash
# Check Python version (need 3.8+)
python --version

# Install dependencies
pip install pandas numpy rich
```

### "Process shows as stale"
The trader process may have stopped. Check logs:
```bash
tail -50 logs/single_model_trial861.log
ps aux | grep paper_trader
```

---

## 🎉 You're All Set!

The dashboard is ready to use. Start monitoring your paper traders and see which strategy performs better!

**Launch command:**
```bash
cd /opt/user-data/experiment/cappuccino
./start_dashboard.sh
```

**Stop dashboard:**
- Press `Ctrl+C`

**Run in background:**
```bash
tmux new -s dashboard
./start_dashboard.sh
# Press Ctrl+B then D to detach
```

---

**Dashboard Status:** ✅ Ready
**Build Time:** ~3 hours
**Lines of Code:** ~540
**Dependencies:** pandas, numpy, rich (optional)
