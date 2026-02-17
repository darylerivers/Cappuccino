# ✓ Docker Setup Complete for Cappuccino

Your Docker environment for the Cappuccino trading system is now fully configured!

## What Was Created

### Core Docker Files

| File | Size | Purpose |
|------|------|---------|
| `Dockerfile` | 1.8K | Container build instructions with CUDA 12.1 and PyTorch |
| `docker-compose.yml` | 2.5K | Multi-service orchestration (training, dashboard, Ollama) |
| `requirements.txt` | 1.1K | Python dependencies (PyTorch, Optuna, pandas, etc.) |
| `.dockerignore` | - | Excludes large files from build context |

### Helper Scripts

| Script | Size | Purpose |
|--------|------|---------|
| `docker_build.sh` | 3.0K | Interactive build script with Ollama model pulling |
| `docker_run.sh` | 4.7K | Interactive launcher with 9 preset training modes |
| `docker_test.sh` | 6.5K | Environment validation (10 comprehensive tests) |
| `Makefile` | 3.6K | Quick commands (make build, make train, etc.) |

### Documentation

| Document | Size | Purpose |
|----------|------|---------|
| `README_DOCKER.md` | 7.8K | Comprehensive Docker documentation |
| `QUICKSTART.md` | 4.8K | 3-step quick start guide |
| `.gitignore` | Updated | Git ignore patterns for Docker artifacts |

## Directory Structure

```
cappuccino/
├── Docker Configuration
│   ├── Dockerfile                 # Container build
│   ├── docker-compose.yml         # Service orchestration
│   ├── requirements.txt           # Python packages
│   └── .dockerignore             # Build exclusions
│
├── Helper Scripts
│   ├── docker_build.sh           # Build with Ollama
│   ├── docker_run.sh             # Interactive launcher
│   ├── docker_test.sh            # Environment tests
│   └── Makefile                  # Make commands
│
├── Documentation
│   ├── README_DOCKER.md          # Full Docker docs
│   ├── QUICKSTART.md             # Quick start
│   └── DOCKER_SETUP_COMPLETE.md  # This file
│
├── Training Scripts
│   ├── 1_optimize_unified.py     # Unified optimizer
│   ├── 0_dl_trainval_data.py     # Data downloader
│   ├── 2_validate.py             # Validator
│   ├── 4_backtest.py             # Backtester
│   └── analyze_training.py       # Results analyzer
│
└── Data Directories (mounted volumes)
    ├── data/                     # Market data
    ├── logs/                     # Training logs
    ├── databases/                # Optuna studies
    ├── train_results/            # Model checkpoints
    └── plots_and_metrics/        # Performance plots
```

## Features

### 🐳 Docker Services

1. **cappuccino-train**
   - NVIDIA CUDA 12.1 with cuDNN 8
   - PyTorch 2.6.0 with GPU support
   - Optuna hyperparameter optimization
   - Volume mounts for data persistence

2. **optuna-dashboard**
   - Real-time study monitoring
   - Access at http://localhost:8080
   - Connected to study databases

3. **ollama**
   - Sentiment analysis models
   - GPU-accelerated inference
   - 5 models: aya, qwen2, phi3, mistral, llama3

### 🚀 Quick Commands

```bash
# Test environment (recommended first step)
./docker_test.sh

# Build container
./docker_build.sh                    # Interactive
make build                           # Quick

# Run training
./docker_run.sh                      # Interactive menu
make train                           # Standard CPCV
make train-multi-tf                  # Multi-timeframe
make train-sentiment                 # With sentiment
make train-rolling                   # Rolling windows

# Monitor
make dashboard                       # Optuna dashboard
make logs-train                      # View logs
make gpu                            # GPU status

# Utilities
make download                        # Get data
make shell                          # Interactive shell
make clean                          # Cleanup
make help                           # All commands
```

### 📋 Training Modes

The `docker_run.sh` script provides 9 preset modes:

1. **Standard CPCV** - 100 trials, combinatorial purged cross-validation
2. **Multi-timeframe** - 150 trials, 5m to 1d timeframes
3. **Rolling windows** - 90/30 day train/test splits
4. **Sentiment analysis** - Uses aya model for market sentiment
5. **Tightened ranges** - 50 trials, exploitation mode
6. **Custom parameters** - Interactive configuration
7. **Interactive shell** - Manual commands
8. **Optuna dashboard** - Study monitoring
9. **Download data** - Market data fetching

## Getting Started (3 Steps)

### Step 1: Test Environment
```bash
cd /home/mrc/experiment/cappuccino
./docker_test.sh
```

This runs 10 tests to verify:
- Docker installation
- Docker Compose
- NVIDIA GPU runtime
- Directory structure
- Environment configuration
- Python imports

### Step 2: Build Container
```bash
./docker_build.sh
```

This will:
- Build the Docker image (~5 min)
- Optionally pull Ollama models (~10 min per model)
- Verify successful build

### Step 3: Run Training
```bash
./docker_run.sh
```

Select your desired training mode from the menu.

## Advanced Usage

### Custom Training Parameters

```bash
# Direct docker-compose
docker-compose run --rm cappuccino-train python 1_optimize_unified.py \
  --mode multi-timeframe \
  --n-trials 200 \
  --gpu 0 \
  --use-sentiment \
  --sentiment-model "mvkvl/sentiments:aya" \
  --study-name my_experiment \
  --storage sqlite:///databases/optuna_my_experiment.db
```

### Multi-GPU Training

```yaml
# Edit docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=0,1
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 2
```

### Parallel Studies

```bash
# Terminal 1 (GPU 0)
docker-compose run -d --name study1 -e CUDA_VISIBLE_DEVICES=0 \
  cappuccino-train python 1_optimize_unified.py --gpu 0 --study-name study1

# Terminal 2 (GPU 1)
docker-compose run -d --name study2 -e CUDA_VISIBLE_DEVICES=1 \
  cappuccino-train python 1_optimize_unified.py --gpu 1 --study-name study2
```

### Development Mode

```bash
# Interactive shell with live code updates
make shell

# Inside container:
python 1_optimize_unified.py --help
python 0_dl_trainval_data.py
pytest -v
```

## Volume Mounts

All important directories are mounted as volumes for persistence:

| Host | Container | Purpose |
|------|-----------|---------|
| `./data` | `/workspace/cappuccino/data` | Market data (OHLCV, sentiment) |
| `./logs` | `/workspace/cappuccino/logs` | Training logs |
| `./databases` | `/workspace/cappuccino/databases` | Optuna databases |
| `./train_results` | `/workspace/cappuccino/train_results` | Model checkpoints |
| `../ghost/FinRL_Crypto` | `/workspace/ghost/FinRL_Crypto` | Parent modules |

Changes to files in these directories persist across container restarts.

## Monitoring & Debugging

### Real-time Logs
```bash
# All services
make logs

# Just training
make logs-train

# Specific log file
docker-compose exec cappuccino-train tail -f logs/training_latest.log
```

### GPU Monitoring
```bash
# Inside container
make gpu

# From host
watch -n 1 nvidia-smi
```

### Optuna Dashboard
```bash
make dashboard
# Visit http://localhost:8080
```

### Database Inspection
```bash
make shell

# Inside container
python -c "
import optuna
study = optuna.load_study(
    study_name='your_study',
    storage='sqlite:///databases/optuna_your_study.db'
)
print(f'Trials: {len(study.trials)}')
print(f'Best trial: {study.best_trial.number}')
print(f'Best value: {study.best_value}')
"
```

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| GPU not detected | Install nvidia-docker2: `sudo apt-get install nvidia-docker2` |
| Permission denied | Add user to docker group: `sudo usermod -aG docker $USER` |
| Out of memory | Reduce batch size or use `--use-best-ranges` |
| Import errors | Verify parent dir mount: `../ghost/FinRL_Crypto` exists |
| Port in use | Change port in docker-compose.yml or stop conflicting service |

### Reset Everything

```bash
# Stop and remove
make down

# Clean volumes
make clean

# Remove images
make clean-all

# Rebuild fresh
make build-no-cache
```

## Performance Tips

1. **Use SSD for volumes** - Significantly faster data loading
2. **Pre-download data** - Run `make download` before training
3. **Use best ranges** - `--use-best-ranges` for faster convergence
4. **Monitor GPU** - Keep utilization >80% for efficiency
5. **Parallel studies** - Use multiple GPUs if available
6. **Cache Ollama models** - Pull once, use for all experiments

## Next Steps

1. **Run your first experiment**
   ```bash
   ./docker_run.sh
   # Select option 1 (Standard CPCV)
   ```

2. **Monitor progress**
   ```bash
   make dashboard
   make logs-train
   ```

3. **Analyze results**
   ```bash
   make shell
   python analyze_training.py
   ```

4. **Iterate and improve**
   - Adjust hyperparameter ranges
   - Try different sentiment models
   - Experiment with timeframes

## Support & Documentation

- **Quick start**: `QUICKSTART.md`
- **Full Docker docs**: `README_DOCKER.md`
- **Make commands**: `make help`
- **Script help**: `./docker_run.sh` (interactive menu)
- **Environment test**: `./docker_test.sh`

---

## Summary

You now have a complete, production-ready Docker environment for the Cappuccino trading system!

**Created:**
- ✓ Docker container with GPU support
- ✓ Multi-service orchestration (training, dashboard, Ollama)
- ✓ Interactive scripts for easy usage
- ✓ Makefile for quick commands
- ✓ Comprehensive documentation
- ✓ Environment testing tools

**Ready to use:**
- ✓ PyTorch 2.6.0 with CUDA 12.1
- ✓ Optuna hyperparameter optimization
- ✓ Sentiment analysis with Ollama
- ✓ Multi-timeframe trading (5m to 1d)
- ✓ Real-time monitoring dashboard

**Next:** Run `./docker_test.sh` to validate your setup, then `./docker_run.sh` to start training!

---

*Cappuccino Docker Setup - Generated 2025-10-27*
