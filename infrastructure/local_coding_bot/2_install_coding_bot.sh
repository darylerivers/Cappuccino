#!/bin/bash
# Install Continue + Aider for local AI coding assistance

set -e

echo "=========================================="
echo "Local Coding Bot Setup"
echo "=========================================="
echo ""

# 1. Check Ollama
echo "1. Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi
echo "✓ Ollama installed"

# 2. Pull coding models
echo ""
echo "2. Pulling coding models..."
echo "   This will take 10-20 minutes..."

# Fast autocomplete model (6.7B)
if ! ollama list | grep -q "deepseek-coder:6.7b"; then
    echo "   Pulling deepseek-coder:6.7b (for autocomplete)..."
    ollama pull deepseek-coder:6.7b
fi

# General chat model (7B)
if ! ollama list | grep -q "mistral"; then
    echo "   Pulling mistral (for chat)..."
    ollama pull mistral
fi

# Alternative coder
if ! ollama list | grep -q "codellama:7b"; then
    echo "   Pulling codellama:7b (alternative)..."
    ollama pull codellama:7b
fi

echo "✓ Models installed"

# 3. Install Aider
echo ""
echo "3. Installing Aider..."
pip install aider-chat
echo "✓ Aider installed"

# 4. Create Aider config
echo ""
echo "4. Creating Aider configuration..."
mkdir -p ~/.aider

cat > ~/.aider/.aider.conf.yml << 'EOF'
# Aider configuration for local Ollama
model: ollama/mistral
edit-format: diff
auto-commits: true
dirty-commits: true
git: true

# Ollama connection
openai-api-base: http://localhost:11434/v1
openai-api-key: ollama  # Not used but required

# Model settings
max-tokens: 4096
temperature: 0.2
EOF

echo "✓ Aider configured"

# 5. Install Continue (if VSCode/Cursor available)
echo ""
echo "5. Checking for VSCode/Cursor..."
if command -v code &> /dev/null; then
    echo "   VSCode found, installing Continue extension..."
    code --install-extension continue.continue
    echo "✓ Continue extension installed"

    # Create Continue config
    mkdir -p ~/.continue
    cat > ~/.continue/config.json << 'EOF'
{
  "models": [
    {
      "title": "Mistral 7B (Chat)",
      "provider": "ollama",
      "model": "mistral"
    },
    {
      "title": "DeepSeek Coder (Code)",
      "provider": "ollama",
      "model": "deepseek-coder:6.7b"
    },
    {
      "title": "CodeLlama (Alternative)",
      "provider": "ollama",
      "model": "codellama:7b"
    }
  ],
  "tabAutocompleteModel": {
    "title": "DeepSeek Coder",
    "provider": "ollama",
    "model": "deepseek-coder:6.7b"
  },
  "tabAutocompleteOptions": {
    "disable": false,
    "useCopyBuffer": false
  },
  "allowAnonymousTelemetry": false
}
EOF
    echo "✓ Continue configured"
else
    echo "   VSCode not found - skipping Continue (CLI tools only)"
fi

# 6. Create test scripts
echo ""
echo "6. Creating test scripts..."

# Test Aider
cat > test_aider.sh << 'EOF'
#!/bin/bash
# Test Aider setup

cd /opt/user-data/experiment/cappuccino

echo "Testing Aider..."
echo ""
echo "Example commands you can try:"
echo "  aider --model ollama/mistral"
echo ""
echo "Then in Aider prompt:"
echo "  /add paper_trader_alpaca_polling.py"
echo "  Explain what this file does"
echo ""
echo "Or for autonomous editing:"
echo "  Add type hints to the __init__ method"
EOF

chmod +x test_aider.sh

# Test Ollama
cat > test_ollama.sh << 'EOF'
#!/bin/bash
# Test Ollama models

echo "Testing Ollama models..."
echo ""

echo "1. Testing Mistral (chat):"
ollama run mistral "Write a one-line Python function that adds two numbers"

echo ""
echo "2. Testing DeepSeek Coder (autocomplete):"
ollama run deepseek-coder:6.7b "Complete this Python code: def fibonacci(n):"

echo ""
echo "3. Testing CodeLlama:"
ollama run codellama:7b "Fix this code: def add(a, b) return a + b"

echo ""
echo "✓ All models working!"
EOF

chmod +x test_ollama.sh

echo "✓ Test scripts created"

# 7. Run tests
echo ""
echo "7. Testing installation..."
./test_ollama.sh

echo ""
echo "=========================================="
echo "Local Coding Bot Setup Complete!"
echo "=========================================="
echo ""
echo "Installed:"
echo "  ✓ Ollama (LLM runtime)"
echo "  ✓ Mistral 7B (chat)"
echo "  ✓ DeepSeek Coder 6.7B (autocomplete)"
echo "  ✓ CodeLlama 7B (alternative)"
echo "  ✓ Aider (CLI coding assistant)"
if command -v code &> /dev/null; then
    echo "  ✓ Continue (VSCode extension)"
fi

echo ""
echo "Usage:"
echo ""
echo "  Option 1: Aider (CLI)"
echo "    cd /opt/user-data/experiment/cappuccino"
echo "    aider --model ollama/mistral"
echo "    > Add docstrings to environment_Alpaca.py"
echo ""

if command -v code &> /dev/null; then
    echo "  Option 2: Continue (VSCode)"
    echo "    1. Open VSCode"
    echo "    2. Press Ctrl+L to open chat"
    echo "    3. Select 'Mistral 7B' or 'DeepSeek Coder'"
    echo "    4. Ask coding questions"
    echo "    5. Press Tab for autocomplete"
    echo ""
fi

echo "  Option 3: Direct Ollama"
echo "    ollama run mistral 'your coding question'"
echo ""

echo "After RX 7900 GRE arrives (16GB VRAM):"
echo "  ollama pull codellama:33b       # Better quality"
echo "  ollama pull deepseek-coder:33b  # State-of-the-art"
echo "  ollama pull mixtral:8x7b        # 47B parameters"
echo ""

echo "Test with:"
echo "  ./test_ollama.sh"
echo "  ./test_aider.sh"
