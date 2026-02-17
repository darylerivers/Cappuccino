FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire cappuccino project
COPY . /workspace/cappuccino/

# Copy parent FinRL_Crypto directory (needed for imports)
# This will be mounted as a volume in docker-compose
# But we create the directory structure here
RUN mkdir -p /workspace/ghost/FinRL_Crypto

# Set Python path to include parent directories
ENV PYTHONPATH="/workspace:/workspace/ghost/FinRL_Crypto:${PYTHONPATH}"

# Create directories for data, logs, databases, and results
RUN mkdir -p /workspace/cappuccino/data \
             /workspace/cappuccino/logs \
             /workspace/cappuccino/databases \
             /workspace/cappuccino/train_results \
             /workspace/cappuccino/plots_and_metrics

# Set working directory to cappuccino
WORKDIR /workspace/cappuccino

# Expose port for Optuna dashboard (optional)
EXPOSE 8080

# Default command - can be overridden in docker-compose or docker run
CMD ["python", "1_optimize_unified.py", "--help"]
