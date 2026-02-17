#!/bin/bash
# Activate the cappuccino-rocm environment with ROCm support
# Source this file before running training: source activate_rocm_env.sh

export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
pyenv activate cappuccino-rocm

# ROCm environment variables
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0

echo "✓ Activated cappuccino-rocm environment"
echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
python -c 'import torch; na="N/A"; print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else na}")'
