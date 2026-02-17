#!/usr/bin/env bash
# Setup script for TinyLlama AI Training System

echo "======================================================================"
echo "  CAPPUCCINO AI TRAINING SYSTEM SETUP"
echo "======================================================================"
echo ""

# Check if running with GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: nvidia-smi not found - GPU may not be available"
    echo "   Training will use CPU (slower but works)"
else
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader
fi

echo ""
echo "Installing dependencies..."
echo ""

# Install Unsloth for optimized training
echo "1. Installing Unsloth (optimized LoRA training)..."
pip install --upgrade "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install required packages
echo "2. Installing required packages..."
pip install torch transformers datasets trl peft accelerate bitsandbytes

echo ""
echo "======================================================================"
echo "  SETUP COMPLETE!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Start data collection:"
echo "   ./start_automation.sh"
echo "   (Let it run to collect trade data)"
echo ""
echo "2. Check collected data:"
echo "   python3 -c 'from ai_training.trade_logger import get_trade_logger; print(get_trade_logger().get_training_stats())'"
echo ""
echo "3. Once you have 50+ trades with outcomes, start training:"
echo "   python3 ai_training/train_tinyllama.py"
echo ""
echo "4. Set up continuous learning (optional):"
echo "   python3 ai_training/continuous_learning.py --schedule weekly"
echo ""
