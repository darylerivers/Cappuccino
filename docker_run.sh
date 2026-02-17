#!/bin/bash
################################################################################
# Docker Run Script for Cappuccino
# Quick launcher for common training scenarios
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================================================"
echo "CAPPUCCINO TRAINING LAUNCHER"
echo "================================================================================"
echo ""
echo "Select training mode:"
echo "  1) Standard CPCV (100 trials)"
echo "  2) Multi-timeframe (150 trials)"
echo "  3) Rolling windows (90/30 day splits)"
echo "  4) With sentiment analysis (aya model)"
echo "  5) Tightened ranges / exploitation (50 trials)"
echo "  6) Custom parameters (interactive)"
echo "  7) Interactive bash shell"
echo "  8) Start Optuna dashboard"
echo "  9) Download data only"
echo ""
read -p "Enter choice [1-9]: " choice

case $choice in
    1)
        echo ""
        echo "${GREEN}Running standard CPCV optimization (100 trials)...${NC}"
        docker-compose run --rm cappuccino-train python scripts/training/1_optimize_unified.py \
            --n-trials 100 \
            --gpu 0 \
            --study-name cappuccino_standard \
            --storage sqlite:///databases/optuna_standard.db
        ;;

    2)
        echo ""
        echo "${GREEN}Running multi-timeframe optimization (150 trials)...${NC}"
        docker-compose run --rm cappuccino-train python scripts/training/1_optimize_unified.py \
            --mode multi-timeframe \
            --n-trials 150 \
            --gpu 0 \
            --study-name cappuccino_multi_tf \
            --storage sqlite:///databases/optuna_multi_tf.db
        ;;

    3)
        echo ""
        echo "${GREEN}Running rolling window optimization (90/30 days)...${NC}"
        docker-compose run --rm cappuccino-train python scripts/training/1_optimize_unified.py \
            --mode rolling \
            --window-train-days 90 \
            --window-test-days 30 \
            --n-trials 100 \
            --gpu 0 \
            --study-name cappuccino_rolling \
            --storage sqlite:///databases/optuna_rolling.db
        ;;

    4)
        echo ""
        echo "${GREEN}Running with sentiment analysis (aya model)...${NC}"
        docker-compose run --rm cappuccino-train python scripts/training/1_optimize_unified.py \
            --use-sentiment \
            --sentiment-model "mvkvl/sentiments:aya" \
            --n-trials 150 \
            --gpu 0 \
            --study-name cappuccino_sentiment_aya \
            --storage sqlite:///databases/optuna_sentiment_aya.db
        ;;

    5)
        echo ""
        echo "${GREEN}Running with tightened ranges (50 trials)...${NC}"
        docker-compose run --rm cappuccino-train python scripts/training/1_optimize_unified.py \
            --use-best-ranges \
            --n-trials 50 \
            --gpu 0 \
            --study-name cappuccino_best_ranges \
            --storage sqlite:///databases/optuna_best_ranges.db
        ;;

    6)
        echo ""
        echo "${YELLOW}Custom parameters mode${NC}"
        echo ""
        read -p "Number of trials: " n_trials
        read -p "GPU ID [0]: " gpu
        gpu=${gpu:-0}
        read -p "Study name: " study_name
        read -p "Mode [standard/multi-timeframe/rolling]: " mode
        mode=${mode:-standard}

        echo ""
        echo "${GREEN}Running with custom parameters...${NC}"
        docker-compose run --rm cappuccino-train python scripts/training/1_optimize_unified.py \
            --mode "$mode" \
            --n-trials "$n_trials" \
            --gpu "$gpu" \
            --study-name "$study_name" \
            --storage "sqlite:///databases/optuna_${study_name}.db"
        ;;

    7)
        echo ""
        echo "${GREEN}Starting interactive bash shell...${NC}"
        echo "Type 'exit' to quit the container"
        echo ""
        docker-compose run --rm cappuccino-train bash
        ;;

    8)
        echo ""
        echo "${GREEN}Starting Optuna dashboard...${NC}"
        echo "Dashboard will be available at http://localhost:8080"
        echo "Press Ctrl+C to stop"
        echo ""
        docker-compose up optuna-dashboard
        ;;

    9)
        echo ""
        echo "${GREEN}Downloading training data...${NC}"
        docker-compose run --rm cappuccino-train python scripts/data/0_dl_trainval_data.py
        ;;

    *)
        echo ""
        echo "${RED}Invalid choice!${NC}"
        exit 1
        ;;
esac

echo ""
echo "================================================================================"
echo "${GREEN}DONE!${NC}"
echo "================================================================================"
