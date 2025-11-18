# Cappuccino Trial #13 - Deployment Roadmap

## 🎯 Optimization Complete!

**Best Configuration**: Trial #13 from `cappuccino_exploitation` study
**Objective Value**: 0.077980 (Sharpe Δ)
**Improvement**: 33% better than initial best (Trial #98: 0.058566)
**Convergence**: 156 subsequent trials failed to beat it

---

## 📋 Deployment Pipeline

### Phase 1: Backtesting (Historical Validation) ✅ READY
**Objective**: Test Trial #13 on historical out-of-sample data

**Steps**:
1. Load Trial #13 trained model from: `train_results/cwd_tests/trial_13_1h/`
2. Test on held-out historical data (post-training period)
3. Calculate performance metrics:
   - Sharpe Ratio (vs HODL)
   - Max Drawdown
   - Win Rate
   - Total Return
   - Volatility

**Script**: Adapt `/ghost/FinRL_Crypto/4_backtest.py`

**Command**:
```bash
cd /home/mrc/experiment/cappuccino
python backtest_trial13.py --model-path train_results/cwd_tests/trial_13_1h/ \
                            --data-path data/1h_test/ \
                            --output results/backtest_trial13.json
```

**Success Criteria**:
- Sharpe Ratio > 0.5
- Max Drawdown < 30%
- Positive total return
- Win rate > 45%

---

###Phase 2: Forward Validation (Recent Data) ✅ READY
**Objective**: Validate on most recent unseen data

**Steps**:
1. Download fresh market data (last 2-4 weeks)
2. Run Trial #13 model on this data
3. Compare with simulated HODL strategy
4. Analyze trade frequency and patterns

**Script**: Adapt `/ghost/FinRL_Crypto/2_validate.py`

**Command**:
```bash
python validate_trial13.py --model-path train_results/cwd_tests/trial_13_1h/ \
                            --start-date 2025-10-01 \
                            --end-date 2025-10-30 \
                            --output results/validation_trial13.json
```

**Success Criteria**:
- Performance consistent with backtesting
- No catastrophic failures
- Reasonable trade patterns (not over-trading)
- Adapts to recent market conditions

---

### Phase 3: Paper Trading Setup 🔧 CONFIGURATION NEEDED
**Objective**: Deploy to paper trading environment for live testing

**Requirements**:
1. **Alpaca Paper Trading Account**:
   - Sign up at: https://alpaca.markets/
   - Get API keys (paper trading)
   - Configure in `config_api.py`

2. **Live Data Feed**:
   - Real-time 1h candles
   - WebSocket or polling connection
   - Symbol: BTC/USD (or configured symbol)

3. **System Requirements**:
   - Keep GPU/CPU running 24/7
   - Internet connection
   - Error monitoring/alerting

**Script**: Use `/ghost/FinRL_Crypto/paper_trader_alpaca_polling.py`

**Configuration**:
```python
# config_api.py
ALPACA_CONFIG = {
    "API_KEY": "your_paper_key",
    "API_SECRET": "your_paper_secret",
    "BASE_URL": "https://paper-api.alpaca.markets",  # Paper trading
}

TRADING_CONFIG = {
    "model_path": "train_results/cwd_tests/trial_13_1h/",
    "symbol": "BTCUSD",
    "timeframe": "1h",
    "initial_capital": 10000,  # Paper trading capital
    "max_position_size": 0.95,  # From Trial #13 config
    "check_interval": 3600,  # 1 hour in seconds
}
```

**Deployment Command**:
```bash
# Start paper trading bot
nohup python paper_trader_alpaca_polling.py \
    --config config_trial13.json \
    --log-file logs/paper_trading.log \
    > paper_trading_output.log 2>&1 &

# Monitor
tail -f logs/paper_trading.log
```

**Monitoring Dashboard**:
- Track P&L daily
- Monitor trade frequency
- Alert on drawdowns > 10%
- Compare with HODL strategy

---

### Phase 4: Live Trading Deployment 🚨 CAUTION
**Objective**: Deploy to live trading with real capital

**Prerequisites** (ALL MUST PASS):
- [ ] Backtest Sharpe > 0.5
- [ ] Forward validation successful
- [ ] Paper trading profitable for 2+ weeks
- [ ] Max drawdown < 20% in paper trading
- [ ] No critical bugs/crashes
- [ ] Risk management validated

**Steps**:
1. Start with SMALL capital (e.g., $500)
2. Use Alpaca live API keys
3. Enable strict risk limits:
   - Max daily loss: 5%
   - Max position size: 90%
   - Emergency stop-loss
4. Monitor CONSTANTLY for first week
5. Gradually increase capital if stable

**Safety Measures**:
- Circuit breakers (halt trading if loss > X%)
- Position size limits
- Maximum daily trades limit
- Heartbeat monitoring
- SMS/email alerts

---

## 🛠️ Next Steps (Immediate Actions)

### Step 1: Create Backtest Script
```bash
# Copy and adapt ghost backtest script
cp /home/mrc/experiment/ghost/FinRL_Crypto/4_backtest.py \
   /home/mrc/experiment/cappuccino/backtest_trial13.py

# Edit to use Trial #13 config
```

### Step 2: Download Test Data
```bash
# Get fresh test data (after training period)
python 0_dl_trade_data.py --start 2023-10-02 --end 2025-10-30 \
                           --output data/1h_test/
```

### Step 3: Run Backtest
```bash
python backtest_trial13.py
```

### Step 4: Analyze Results
- Review backtest metrics
- Compare vs HODL
- Check for overfitting signs
- Validate risk metrics

---

## 📊 Expected Performance (Based on Trial #13)

| Metric | Expected Range | Source |
|--------|---------------|--------|
| Sharpe Ratio (vs HODL) | +0.070 to +0.080 | Trial #13 optimization |
| Win Rate | 50-55% | Typical for DRL strategies |
| Max Drawdown | 20-30% | Crypto market volatility |
| Avg Trade Duration | 6-12 hours | 1h timeframe, ~6-12 candles |
| Trades per Day | 2-4 | Based on 1h trading frequency |

---

## 🔧 Trial #13 Configuration Reference

See `TRIAL_13_BEST_CONFIG.py` for complete hyperparameters.

**Key Parameters**:
- Learning Rate: 1.33e-6 (very stable)
- Batch Size: 3072 (large, stable gradients)
- Gamma: 0.98 (high future reward weight)
- Network Size: 1536 dimensions
- Risk Management: 5% cash reserve, 5% concentration penalty

---

## 📁 File Structure

```
cappuccino/
├── TRIAL_13_BEST_CONFIG.py       # Production hyperparameters
├── DEPLOYMENT_ROADMAP.md          # This file
├── backtest_trial13.py            # Backtesting script (to create)
├── validate_trial13.py            # Validation script (to create)
├── paper_trader_trial13.py        # Paper trading bot (to create)
├── train_results/
│   └── cwd_tests/
│       └── trial_13_1h/          # Trained model checkpoints
├── results/
│   ├── backtest_trial13.json     # Backtest results
│   └── validation_trial13.json   # Validation results
└── logs/
    └── paper_trading.log          # Live trading logs
```

---

## ⚠️ Risk Warnings

1. **Past performance ≠ Future results**
   - Trial #13 optimized on 2023 data
   - Market conditions change
   - Continuous monitoring required

2. **Overfitting Risk**
   - 170+ trials may have overfit to training data
   - Backtest on truly out-of-sample data essential
   - Watch for performance degradation

3. **Crypto Volatility**
   - High volatility = high risk
   - Drawdowns can be severe
   - Never risk more than you can lose

4. **System Risks**
   - Internet outages
   - API failures
   - GPU/system crashes
   - Exchange downtime

---

## 📞 Monitoring Checklist

**Daily**:
- [ ] Check P&L
- [ ] Review trades
- [ ] Monitor drawdown
- [ ] Check system health

**Weekly**:
- [ ] Compare vs HODL
- [ ] Analyze trade patterns
- [ ] Review risk metrics
- [ ] Update performance log

**Monthly**:
- [ ] Full performance review
- [ ] Consider retraining
- [ ] Adjust risk parameters
- [ ] Evaluate continuation

---

## 🚀 Ready to Deploy?

**Current Status**: ✅ Optimization Complete
**Next Action**: Backtest Trial #13 on out-of-sample data

**Command to start**:
```bash
cd /home/mrc/experiment/cappuccino
# Create backtest script first, then run it
```

---

*Last Updated: 2025-10-30*
*Trial #13 Objective: 0.077980*
*Status: READY FOR BACKTESTING*
