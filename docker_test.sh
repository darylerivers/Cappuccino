#!/bin/bash
################################################################################
# Docker Test Script for Cappuccino
# Verifies that the Docker environment is set up correctly
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS="${GREEN}✓${NC}"
FAIL="${RED}✗${NC}"
WARN="${YELLOW}⚠${NC}"

echo "================================================================================"
echo "CAPPUCCINO DOCKER ENVIRONMENT TEST"
echo "================================================================================"
echo ""

# Test 1: Docker installation
echo -n "1. Checking Docker installation... "
if command -v docker &> /dev/null; then
    echo -e "$PASS"
    docker --version
else
    echo -e "$FAIL"
    echo "   Docker is not installed or not in PATH"
    exit 1
fi
echo ""

# Test 2: Docker Compose installation
echo -n "2. Checking Docker Compose installation... "
if command -v docker-compose &> /dev/null; then
    echo -e "$PASS"
    docker-compose --version
else
    echo -e "$FAIL"
    echo "   Docker Compose is not installed or not in PATH"
    exit 1
fi
echo ""

# Test 3: Docker daemon running
echo -n "3. Checking Docker daemon... "
if docker info &> /dev/null; then
    echo -e "$PASS"
else
    echo -e "$FAIL"
    echo "   Docker daemon is not running"
    echo "   Try: sudo systemctl start docker"
    exit 1
fi
echo ""

# Test 4: NVIDIA Docker runtime
echo -n "4. Checking NVIDIA Docker runtime... "
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo -e "$PASS"
    echo "   GPU detected:"
    docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    echo -e "$WARN"
    echo "   NVIDIA Docker runtime not working"
    echo "   GPU training will not be available"
    echo "   Install nvidia-docker2 if you have an NVIDIA GPU"
fi
echo ""

# Test 5: Image exists
echo -n "5. Checking Cappuccino Docker image... "
if docker images | grep -q "cappuccino"; then
    echo -e "$PASS"
    docker images cappuccino:latest --format "   {{.Repository}}:{{.Tag}} ({{.Size}})"
else
    echo -e "$WARN"
    echo "   Image not found. Run: ./docker_build.sh or make build"
fi
echo ""

# Test 6: Required directories
echo "6. Checking directory structure... "
DIRS=("data" "logs" "databases" "train_results")
all_dirs_ok=true

for dir in "${DIRS[@]}"; do
    echo -n "   - $dir/: "
    if [ -d "$dir" ]; then
        echo -e "$PASS"
    else
        echo -e "$WARN (will be created)"
        all_dirs_ok=false
    fi
done

if [ "$all_dirs_ok" = false ]; then
    echo ""
    echo "   Creating missing directories..."
    mkdir -p data logs databases train_results plots_and_metrics
    echo -e "   $PASS Directories created"
fi
echo ""

# Test 7: Environment file
echo -n "7. Checking .env file... "
if [ -f ".env" ]; then
    echo -e "$PASS"
    if grep -q "ALPACA_API_KEY=your_key_here" .env || grep -q "ALPACA_API_KEY=$" .env; then
        echo -e "   $WARN .env file exists but may need configuration"
        echo "   Edit .env and add your API keys"
    else
        echo "   .env file configured"
    fi
else
    echo -e "$WARN"
    echo "   .env file not found. Copy from .env.template:"
    echo "   cp .env.template .env"
fi
echo ""

# Test 8: Parent directory access
echo -n "8. Checking parent FinRL_Crypto directory... "
if [ -d "../ghost/FinRL_Crypto" ]; then
    echo -e "$PASS"
    echo "   $(ls -1 ../ghost/FinRL_Crypto | grep -E '\.py$|^drl_agents$|^environment' | wc -l) Python files/modules found"
else
    echo -e "$FAIL"
    echo "   Parent directory not found at ../ghost/FinRL_Crypto"
    echo "   Required for imports. Check directory structure."
    exit 1
fi
echo ""

# Test 9: Docker build test (if image exists)
if docker images | grep -q "cappuccino"; then
    echo "9. Testing container startup... "
    if timeout 30 docker-compose run --rm cappuccino-train python --version &> /dev/null; then
        echo -e "   $PASS Container starts successfully"
        version=$(docker-compose run --rm cappuccino-train python --version 2>&1 | tr -d '\r')
        echo "   $version"
    else
        echo -e "   $FAIL Container failed to start"
        echo "   Try rebuilding: make build-no-cache"
    fi
    echo ""

    # Test 10: Python imports
    echo "10. Testing Python imports... "
    imports_test="
import sys
try:
    import torch; print(f'  - PyTorch {torch.__version__}')
    import optuna; print(f'  - Optuna {optuna.__version__}')
    import pandas; print(f'  - Pandas {pandas.__version__}')
    import numpy; print(f'  - NumPy {numpy.__version__}')
    print('All imports successful')
except ImportError as e:
    print(f'Import error: {e}')
    sys.exit(1)
"
    if docker-compose run --rm cappuccino-train python -c "$imports_test" 2>&1 | grep -q "All imports successful"; then
        echo -e "   $PASS"
        docker-compose run --rm cappuccino-train python -c "$imports_test" 2>&1 | grep -E "^  -|successful"
    else
        echo -e "   $FAIL Some imports failed"
        docker-compose run --rm cappuccino-train python -c "$imports_test"
    fi
    echo ""
else
    echo "9-10. Skipping container tests (image not built)"
    echo "      Run ./docker_build.sh to build the image first"
    echo ""
fi

echo "================================================================================"
echo "TEST SUMMARY"
echo "================================================================================"
echo ""

if command -v docker &> /dev/null && \
   command -v docker-compose &> /dev/null && \
   docker info &> /dev/null && \
   [ -d "../ghost/FinRL_Crypto" ]; then
    echo -e "${GREEN}✓ Core requirements met!${NC}"
    echo ""
    if docker images | grep -q "cappuccino"; then
        echo "Next steps:"
        echo "  1. Configure .env file with your API keys"
        echo "  2. Download data: make download"
        echo "  3. Start training: ./docker_run.sh"
        echo ""
        echo "Or run: make help"
    else
        echo "Next steps:"
        echo "  1. Build the image: ./docker_build.sh"
        echo "  2. Configure .env file"
        echo "  3. Start training: ./docker_run.sh"
    fi
else
    echo -e "${RED}✗ Some requirements are missing${NC}"
    echo ""
    echo "Please fix the issues above and run this test again."
    exit 1
fi

echo "================================================================================"
