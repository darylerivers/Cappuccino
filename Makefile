.PHONY: help build up down restart logs shell train dashboard clean test

# Default target
help:
	@echo "Cappuccino Docker Commands"
	@echo "=========================="
	@echo ""
	@echo "Setup:"
	@echo "  make build          Build Docker image"
	@echo "  make build-no-cache Build without cache (clean build)"
	@echo ""
	@echo "Running:"
	@echo "  make up             Start all services"
	@echo "  make down           Stop all services"
	@echo "  make restart        Restart all services"
	@echo "  make train          Run training with default parameters"
	@echo "  make shell          Open interactive bash shell"
	@echo ""
	@echo "Monitoring:"
	@echo "  make logs           View logs (all services)"
	@echo "  make logs-train     View training logs only"
	@echo "  make dashboard      Start Optuna dashboard"
	@echo "  make gpu            Check GPU status"
	@echo ""
	@echo "Data:"
	@echo "  make download       Download training data"
	@echo "  make validate       Run validation"
	@echo "  make backtest       Run backtest"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          Clean up containers and volumes"
	@echo "  make clean-all      Clean everything including images"
	@echo "  make test           Run tests"
	@echo ""

# Build commands
build:
	docker-compose build

build-no-cache:
	docker-compose build --no-cache

# Service management
up:
	docker-compose up -d

down:
	docker-compose down

restart:
	docker-compose restart

# Training commands
train:
	docker-compose run --rm cappuccino-train python 1_optimize_unified.py \
		--n-trials 100 \
		--gpu 0 \
		--study-name cappuccino_default \
		--storage sqlite:///databases/optuna_default.db

train-multi-tf:
	docker-compose run --rm cappuccino-train python 1_optimize_unified.py \
		--mode multi-timeframe \
		--n-trials 150 \
		--gpu 0 \
		--study-name cappuccino_multi_tf \
		--storage sqlite:///databases/optuna_multi_tf.db

train-sentiment:
	docker-compose run --rm cappuccino-train python 1_optimize_unified.py \
		--use-sentiment \
		--sentiment-model "mvkvl/sentiments:aya" \
		--n-trials 150 \
		--gpu 0 \
		--study-name cappuccino_sentiment \
		--storage sqlite:///databases/optuna_sentiment.db

train-rolling:
	docker-compose run --rm cappuccino-train python 1_optimize_unified.py \
		--mode rolling \
		--window-train-days 90 \
		--window-test-days 30 \
		--n-trials 100 \
		--gpu 0 \
		--study-name cappuccino_rolling \
		--storage sqlite:///databases/optuna_rolling.db

# Shell access
shell:
	docker-compose run --rm cappuccino-train bash

# Monitoring
logs:
	docker-compose logs -f

logs-train:
	docker-compose logs -f cappuccino-train

dashboard:
	docker-compose up -d optuna-dashboard
	@echo "Dashboard available at http://localhost:8080"

gpu:
	docker-compose exec cappuccino-train nvidia-smi

# Data operations
download:
	docker-compose run --rm cappuccino-train python 0_dl_trainval_data.py

validate:
	docker-compose run --rm cappuccino-train python 2_validate.py

backtest:
	docker-compose run --rm cappuccino-train python 4_backtest.py

# Cleanup
clean:
	docker-compose down -v
	rm -rf __pycache__ .pytest_cache

clean-all: clean
	docker rmi cappuccino:latest || true
	docker system prune -f

# Testing
test:
	docker-compose run --rm cappuccino-train pytest -v

# Pull Ollama models
pull-ollama-models:
	docker-compose up -d ollama
	@sleep 5
	docker-compose exec ollama ollama pull mvkvl/sentiments:aya
	docker-compose exec ollama ollama pull mvkvl/sentiments:qwen2
	docker-compose exec ollama ollama pull mvkvl/sentiments:phi3
	docker-compose exec ollama ollama pull mvkvl/sentiments:mistral
	docker-compose exec ollama ollama pull mvkvl/sentiments:llama3
