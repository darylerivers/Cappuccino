# Cappuccino Docker Setup

This directory contains Docker configuration for the Cappuccino cryptocurrency trading system with reinforcement learning and sentiment analysis.

## Prerequisites

1. **NVIDIA Docker Runtime** (for GPU support):
   ```bash
   # Install nvidia-docker2
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   sudo apt-get update
   sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

2. **Docker Compose** (usually included with Docker Desktop):
   ```bash
   sudo apt-get install docker-compose-plugin
   ```

## Quick Start

### 1. Build the Container

```bash
cd /home/mrc/experiment/cappuccino
docker-compose build
```

### 2. Run Training

```bash
# Run with default settings
docker-compose up cappuccino-train

# Run in detached mode (background)
docker-compose up -d cappuccino-train

# Run with custom parameters
docker-compose run --rm cappuccino-train python 1_optimize_unified.py \
  --mode multi-timeframe \
  --n-trials 150 \
  --gpu 0 \
  --study-name my_experiment
```

### 3. Monitor with Optuna Dashboard

```bash
# Start the dashboard
docker-compose up -d optuna-dashboard

# Access at http://localhost:8080
```

### 4. Interactive Shell

```bash
# Enter container for manual commands
docker-compose run --rm cappuccino-train bash

# Inside container:
python 1_optimize_unified.py --help
python 0_dl_trainval_data.py
```

## Service Descriptions

### cappuccino-train
Main training service that runs the unified optimization script. Uses GPU for PPO training.

**Key Features:**
- CUDA 12.1 with cuDNN 8
- PyTorch 2.6.0 with GPU support
- Optuna hyperparameter optimization
- Volume mounts for data persistence

### optuna-dashboard
Web dashboard for monitoring Optuna studies in real-time.

**Access:** http://localhost:8080

### ollama
Sentiment analysis service using Ollama models (aya, qwen2, phi3, mistral, llama3).

**Port:** 11434

## Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./data` | `/workspace/cappuccino/data` | Market data (OHLCV, sentiment) |
| `./logs` | `/workspace/cappuccino/logs` | Training logs |
| `./databases` | `/workspace/cappuccino/databases` | Optuna study databases |
| `./train_results` | `/workspace/cappuccino/train_results` | Model checkpoints and metrics |
| `../ghost/FinRL_Crypto` | `/workspace/ghost/FinRL_Crypto` | Parent modules (agents, env) |

## Common Commands

### Training Modes

```bash
# Standard CPCV optimization
docker-compose run --rm cappuccino-train python 1_optimize_unified.py \
  --n-trials 100 --gpu 0

# Multi-timeframe optimization
docker-compose run --rm cappuccino-train python 1_optimize_unified.py \
  --mode multi-timeframe --n-trials 150

# Rolling window optimization
docker-compose run --rm cappuccino-train python 1_optimize_unified.py \
  --mode rolling --window-train-days 90 --window-test-days 30

# With sentiment analysis
docker-compose run --rm cappuccino-train python 1_optimize_unified.py \
  --use-sentiment --sentiment-model "mvkvl/sentiments:aya"

# Tightened ranges (exploitation)
docker-compose run --rm cappuccino-train python 1_optimize_unified.py \
  --use-best-ranges --n-trials 50
```

### Data Management

```bash
# Download training data
docker-compose run --rm cappuccino-train python 0_dl_trainval_data.py

# Validate results
docker-compose run --rm cappuccino-train python 2_validate.py --trial 41

# Run backtest
docker-compose run --rm cappuccino-train python 4_backtest.py
```

### Monitoring

```bash
# View logs in real-time
docker-compose logs -f cappuccino-train

# Check GPU usage inside container
docker-compose exec cappuccino-train nvidia-smi

# Monitor specific log file
docker-compose exec cappuccino-train tail -f logs/training_latest.log
```

### Database Management

```bash
# List Optuna studies
docker-compose run --rm cappuccino-train python -c "
import optuna
study = optuna.load_study(
    study_name='cappuccino_trial',
    storage='sqlite:///databases/optuna_cappuccino.db'
)
print(f'Best trial: {study.best_trial.number}')
print(f'Best value: {study.best_value}')
"

# Export study results
docker-compose run --rm cappuccino-train python analyze_training.py
```

## Environment Variables

Create a `.env` file in the cappuccino directory:

```bash
# API Keys
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Ollama Configuration
OLLAMA_HOST=http://ollama:11434

# Training Configuration
DEFAULT_GPU=0
DEFAULT_TRIALS=100
```

## Troubleshooting

### GPU Not Detected

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If fails, check nvidia-docker installation
sudo systemctl status docker
```

### Out of Memory

```bash
# Reduce batch size in hyperparameters
# Or limit GPU memory in docker-compose.yml:
deploy:
  resources:
    reservations:
      devices:
        - capabilities: [gpu]
          device_ids: ['0']
          options:
            memory: 4GB  # Limit to 4GB
```

### Import Errors

The container needs access to the parent `FinRL_Crypto` directory. Ensure the volume mount is correct:

```yaml
volumes:
  - ../ghost/FinRL_Crypto:/workspace/ghost/FinRL_Crypto
```

### Permission Issues

```bash
# Fix ownership of generated files
sudo chown -R $USER:$USER data/ logs/ databases/ train_results/
```

## Production Deployment

### Using Multiple GPUs

```yaml
# In docker-compose.yml, modify:
environment:
  - CUDA_VISIBLE_DEVICES=0,1  # Use GPUs 0 and 1

deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 2  # Request 2 GPUs
```

### Running Multiple Studies in Parallel

```bash
# GPU 0
docker-compose run -d --name study1 -e CUDA_VISIBLE_DEVICES=0 \
  cappuccino-train python 1_optimize_unified.py --gpu 0 --study-name study1

# GPU 1
docker-compose run -d --name study2 -e CUDA_VISIBLE_DEVICES=1 \
  cappuccino-train python 1_optimize_unified.py --gpu 1 --study-name study2
```

### Persistent Containers

```bash
# Start and keep running
docker-compose up -d

# Execute commands in running container
docker-compose exec cappuccino-train python 1_optimize_unified.py

# Stop all services
docker-compose down
```

## File Structure

```
cappuccino/
├── Dockerfile              # Container build instructions
├── docker-compose.yml      # Multi-service orchestration
├── requirements.txt        # Python dependencies
├── .dockerignore          # Files to exclude from image
├── .env                   # Environment variables (gitignored)
├── 1_optimize_unified.py  # Main training script
├── data/                  # Market data (mounted)
├── logs/                  # Training logs (mounted)
├── databases/             # Optuna studies (mounted)
└── train_results/         # Model checkpoints (mounted)
```

## Updates and Maintenance

### Rebuild After Code Changes

```bash
# Rebuild image
docker-compose build --no-cache

# Restart services
docker-compose restart
```

### Update Dependencies

```bash
# Edit requirements.txt, then:
docker-compose build

# Or update in running container:
docker-compose exec cappuccino-train pip install -U package_name
```

### Clean Up

```bash
# Remove containers
docker-compose down

# Remove containers and volumes
docker-compose down -v

# Remove images
docker rmi cappuccino:latest

# Full cleanup (careful!)
docker system prune -a
```

## Support

For issues specific to:
- **Docker setup**: Check this README
- **Training scripts**: See main project README
- **GPU issues**: Verify NVIDIA drivers and nvidia-docker

---

Generated for Cappuccino Trading System - Multi-timeframe RL with Sentiment Analysis
