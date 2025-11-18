# Cappuccino Quick Start Guide

Get started with Cappuccino in 3 simple steps.

## Prerequisites

- Docker with NVIDIA GPU support
- NVIDIA GPU with CUDA capability
- At least 8GB GPU VRAM recommended

## Step 1: Build the Container

```bash
cd /home/mrc/experiment/cappuccino

# Option A: Use the build script (recommended)
./docker_build.sh

# Option B: Use Makefile
make build

# Option C: Use docker-compose directly
docker-compose build
```

## Step 2: Configure Environment

```bash
# Copy template and edit with your API keys
cp .env.template .env
nano .env  # or vim, code, etc.
```

Add your Alpaca API credentials:
```bash
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

## Step 3: Run Training

### Easy Mode (Interactive Menu)
```bash
./docker_run.sh
```

Select from:
1. Standard CPCV (100 trials)
2. Multi-timeframe (150 trials)
3. Rolling windows (90/30 days)
4. With sentiment analysis
5. Tightened ranges (50 trials)
6. Custom parameters
7. Interactive shell
8. Optuna dashboard
9. Download data

### Using Makefile (Quick Commands)

```bash
# Download data first
make download

# Run default training
make train

# Run multi-timeframe
make train-multi-tf

# Run with sentiment
make train-sentiment

# Run rolling windows
make train-rolling

# Start Optuna dashboard
make dashboard  # Then visit http://localhost:8080

# Check GPU usage
make gpu

# View logs
make logs-train

# Interactive shell
make shell
```

### Using Docker Compose Directly

```bash
# Standard training
docker-compose run --rm cappuccino-train python 1_optimize_unified.py \
  --n-trials 100 \
  --gpu 0

# Multi-timeframe
docker-compose run --rm cappuccino-train python 1_optimize_unified.py \
  --mode multi-timeframe \
  --n-trials 150

# With sentiment
docker-compose run --rm cappuccino-train python 1_optimize_unified.py \
  --use-sentiment \
  --sentiment-model "mvkvl/sentiments:aya"
```

## Monitoring

### Real-time Logs
```bash
# All logs
make logs

# Just training
make logs-train

# Or using docker-compose
docker-compose logs -f cappuccino-train
```

### Optuna Dashboard
```bash
# Start dashboard
make dashboard

# Or
docker-compose up -d optuna-dashboard

# Visit http://localhost:8080
```

### GPU Monitoring
```bash
# Inside container
make gpu

# Or from host
watch -n 1 nvidia-smi
```

## Results

Results are saved to mounted volumes:

```
cappuccino/
├── train_results/     # Model checkpoints
│   └── res_YYYY-MM-DD__HH_MM_SS_study_name/
│       ├── model.zip
│       ├── metrics.csv
│       └── config.json
├── logs/              # Training logs
│   └── training_*.log
├── databases/         # Optuna databases
│   └── optuna_*.db
└── plots_and_metrics/ # Performance plots
    ├── cumulative_return.png
    └── metrics.txt
```

## Common Tasks

### Download Market Data
```bash
make download
# Or
docker-compose run --rm cappuccino-train python 0_dl_trainval_data.py
```

### Validate a Trial
```bash
docker-compose run --rm cappuccino-train python 2_validate.py --trial 41
```

### Run Backtest
```bash
make backtest
# Or
docker-compose run --rm cappuccino-train python 4_backtest.py
```

### Analyze Results
```bash
docker-compose run --rm cappuccino-train python analyze_training.py
```

## Troubleshooting

### GPU Not Found
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If fails, reinstall nvidia-docker
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Out of Memory
- Reduce `--n-trials` to train fewer trials at once
- Use `--use-best-ranges` for smaller hyperparameter search space
- Reduce batch size in the optimization script

### Import Errors
Ensure parent directory is mounted:
```yaml
# In docker-compose.yml
volumes:
  - ../ghost/FinRL_Crypto:/workspace/ghost/FinRL_Crypto
```

### Permission Errors
```bash
# Fix ownership
sudo chown -R $USER:$USER data/ logs/ databases/ train_results/
```

## Cleaning Up

```bash
# Stop containers
make down

# Remove containers and volumes
make clean

# Full cleanup (including images)
make clean-all
```

## Next Steps

1. **Experiment with hyperparameters**: Edit ranges in `1_optimize_unified.py`
2. **Add custom indicators**: Modify data loading functions
3. **Try different sentiment models**: Use qwen2, phi3, mistral, llama3
4. **Multi-GPU training**: Update docker-compose.yml to use multiple GPUs
5. **Production deployment**: Set up persistent containers with monitoring

## Getting Help

- **Docker issues**: See `README_DOCKER.md`
- **Training configuration**: Run `docker-compose run --rm cappuccino-train python 1_optimize_unified.py --help`
- **Makefile commands**: Run `make help`

---

Happy training! 🚀
