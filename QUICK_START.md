# Quick Start - Organized Cappuccino

## 📍 You Are Here
Your files are now organized! Everything is in logical folders.

## 🎯 Common Tasks

### Check Training Status
```bash
./monitoring/check_status.sh
```

### Monitor Training (Live)
```bash
./monitoring/monitor_training.sh
tail -f logs/training/training_14indicators_*.log
```

### Download New Data
```bash
python scripts/data/0_dl_trainval_data.py
```

### Start Training
```bash
python scripts/training/1_optimize_unified.py
# or automated:
./scripts/automation/start_training.sh
```

### Deploy Model
```bash
python scripts/deployment/auto_model_deployer.py
```

### Check Paper Trader
```bash
python monitoring/show_current_trading_status.py
```

## 📂 Where Is Everything?

### Scripts
- **Training**: `scripts/training/`
- **Data**: `scripts/data/`
- **Deployment**: `scripts/deployment/`
- **Automation**: `scripts/automation/`

### Logs
- **Training**: `logs/training/`
- **System**: `logs/system/`

### Documentation
- **All guides**: `docs/guides/`
- **Status**: `TRAINING_STATUS_14INDICATORS.md`
- **Structure**: `DIRECTORY_STRUCTURE.md`

### Key Files
- **Config**: `config_main.py` (root)
- **Processor**: `processors/processor_Alpaca.py`
- **Monitor**: `monitoring/check_status.sh`

## 🔍 Troubleshooting

### Training Issues
1. Check: `logs/training/*.log`
2. Config: `config_main.py`
3. Monitor: `./monitoring/monitor_training.sh`

### Data Issues
1. Script: `scripts/data/0_dl_trainval_data.py`
2. Processor: `processors/processor_Alpaca.py`
3. Check: `data/1h_1680/`

### Deployment Issues
1. Script: `scripts/deployment/auto_model_deployer.py`
2. Logs: `logs/system/*.log`
3. Status: `./monitoring/check_status.sh`

## 📖 Full Documentation
See: `DIRECTORY_STRUCTURE.md` for complete file organization guide.

---
**Organization Date**: 2026-02-05
**Files Organized**: 2,813 files across 30+ folders
