#!/bin/bash
# Fix Ollama to use AMD GPU (RX 7900 GRE) with ROCm

echo "🔧 Configuring Ollama to use AMD GPU with ROCm..."

# Create systemd override directory
sudo mkdir -p /etc/systemd/system/ollama.service.d

# Create environment override file
sudo tee /etc/systemd/system/ollama.service.d/rocm.conf > /dev/null <<'EOF'
[Service]
Environment="HSA_OVERRIDE_GFX_VERSION=11.0.0"
Environment="HIP_VISIBLE_DEVICES=0"
Environment="ROCR_VISIBLE_DEVICES=0"
Environment="GPU_DEVICE_ORDINAL=0"
Environment="OLLAMA_DEBUG=1"
EOF

echo "✅ Created systemd override with ROCm environment variables"

# Reload systemd and restart Ollama
echo "🔄 Restarting Ollama service..."
sudo systemctl daemon-reload
sudo systemctl restart ollama

echo "⏳ Waiting for Ollama to start..."
sleep 5

# Verify GPU detection
echo ""
echo "🔍 Checking GPU detection..."
journalctl -u ollama --no-pager -n 50 | grep -i "gpu\|rocm\|hip\|amd"

echo ""
echo "📊 Current model status:"
ollama ps

echo ""
echo "✅ Done! Now test with:"
echo "   ollama run glm4:latest 'Hello'"
echo ""
echo "Expected: Should show 'PROCESSOR: 100% GPU' instead of CPU"
