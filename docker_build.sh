#!/bin/bash
################################################################################
# Docker Build Script for Cappuccino
# Builds the Docker image and optionally pulls Ollama models
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "CAPPUCCINO DOCKER BUILD"
echo "================================================================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Check for NVIDIA Docker
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: NVIDIA Docker runtime not detected or not working."
    echo "   GPU support may not be available."
    echo "   Install nvidia-docker2 for GPU support."
    echo ""
fi

# Build the image
echo "Building Docker image..."
echo ""
docker-compose build --no-cache

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Docker image built successfully!"
else
    echo ""
    echo "❌ Docker build failed!"
    exit 1
fi

# Ask if user wants to pull Ollama models
echo ""
echo "================================================================================"
read -p "Do you want to pull Ollama sentiment models? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting Ollama service and pulling models..."
    echo "This may take a while (models are several GB each)..."
    echo ""

    # Start Ollama service
    docker-compose up -d ollama
    sleep 5

    # Pull each model
    MODELS=("mvkvl/sentiments:aya" "mvkvl/sentiments:qwen2" "mvkvl/sentiments:phi3" "mvkvl/sentiments:mistral" "mvkvl/sentiments:llama3")

    for model in "${MODELS[@]}"; do
        echo ""
        echo "Pulling $model..."
        docker-compose exec ollama ollama pull "$model"
    done

    echo ""
    echo "✓ All models pulled successfully!"
fi

echo ""
echo "================================================================================"
echo "BUILD COMPLETE!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Copy .env.template to .env and configure your API keys"
echo "  2. Run training:"
echo "       docker-compose up cappuccino-train"
echo ""
echo "  3. Or run interactively:"
echo "       docker-compose run --rm cappuccino-train bash"
echo ""
echo "  4. Start Optuna dashboard:"
echo "       docker-compose up -d optuna-dashboard"
echo "       Open http://localhost:8080"
echo ""
echo "See README_DOCKER.md for more usage examples."
echo "================================================================================"
