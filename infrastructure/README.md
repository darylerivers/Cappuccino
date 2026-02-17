# Infrastructure Setup - RX 7900 GRE, Tiburtina, & Local Coding Bot

**Timeline: 5 Days Until GPU Arrival**

This infrastructure prepares your system for:
1. **RX 7900 GRE** (16GB VRAM) - 10x parallel workers, 4-5x faster training
2. **Tiburtina Integration** - Local LLM sentiment analysis for trading
3. **Local Coding Bot** - AI-powered coding assistance (free, no API costs)

---

## 📅 5-Day Implementation Timeline

### **Day 1 (Today): Pre-Migration Prep** ⏱️ 30 minutes

**Goal:** Backup everything and prepare infrastructure

```bash
cd /opt/user-data/experiment/cappuccino

# 1. Run pre-migration checklist
chmod +x infrastructure/amd_migration/*.sh
./infrastructure/amd_migration/1_pre_migration_checklist.sh

# 2. Review backup
ls -lh backups/pre_amd_migration/

# 3. Keep Trial #250 running for now
# (It's live with $500 paper trading)
```

**Status:** ✅ System backed up, ready for migration

---

### **Day 2-4: Setup Tiburtina & Coding Bot** ⏱️ 1-2 hours

**Goal:** Get AI infrastructure running (independent of GPU swap)

```bash
# Setup Tiburtina sentiment analysis
cd /opt/user-data/experiment/cappuccino
chmod +x infrastructure/tiburtina_integration/*.sh
./infrastructure/tiburtina_integration/1_setup_tiburtina.sh

# Setup local coding bot
chmod +x infrastructure/local_coding_bot/*.sh
./infrastructure/local_coding_bot/2_install_coding_bot.sh

# Test both
python3 test_sentiment_paper_trader.py
./test_ollama.sh
```

**What you'll have:**
- ✅ Ollama running (local LLM server)
- ✅ Mistral 7B for sentiment analysis
- ✅ DeepSeek Coder for coding assistance
- ✅ Aider CLI tool
- ✅ Continue VSCode extension (if using VSCode)

**Status:** 🚀 AI infrastructure ready

---

### **Day 5: GPU Swap Day** ⏱️ 2-3 hours

**Goal:** Install RX 7900 GRE and migrate to ROCm

#### **Morning: Hardware Swap** (30 min)

```bash
# Stop all processes
pkill -f "paper_trader"
pkill -f "1_optimize_unified"

# Shutdown
sudo shutdown -h now

# Physical swap:
# 1. Remove RTX 3070
# 2. Install RX 7900 GRE
# 3. Boot up
```

#### **Afternoon: Software Setup** (2 hours)

```bash
cd /opt/user-data/experiment/cappuccino

# Step 1: Install ROCm (20 min)
./infrastructure/amd_migration/2_install_rocm.sh
# System will reboot

# Step 2: Install PyTorch ROCm (30 min)
./infrastructure/amd_migration/3_install_pytorch_rocm.sh

# Step 3: Verify setup (10 min)
./infrastructure/amd_migration/4_verify_amd_setup.sh

# Step 4: Update training config (5 min)
./infrastructure/amd_migration/5_update_training_config.sh

# Step 5: Restart paper trader
ps aux | grep paper_trader  # Get PID of old trader
# It should already be using Trial #250 - just verify it's running

# Step 6: Start HIGH PERFORMANCE training!
./start_training_amd.sh
# This starts 10 parallel workers (vs 1 before)

# Step 7: Monitor
./monitor_amd.sh
```

**Expected results:**
- ✅ 10 workers running in parallel
- ✅ ~60-80% trial success rate (vs 1% before)
- ✅ 4-5x faster trial completion
- ✅ VRAM: 13-15GB used (safe headroom)

**Status:** 🔥 Full performance unlocked

---

## 📊 Expected Performance Improvements

| Metric | RTX 3070 (8GB) | RX 7900 GRE (16GB) | Improvement |
|--------|----------------|---------------------|-------------|
| **VRAM** | 8GB | 16GB | **2x** |
| **Workers** | 1 | 10 | **10x** |
| **Trial Success Rate** | 1% | 60-80% | **60-80x** |
| **Trials/Day** | 6-8 | 80-120 | **12-15x** |
| **Batch Size** | 32K max | 98K max | **3x** |
| **Time to Deploy** | Weeks | Days | **10x faster** |

---

## 🛠️ Directory Structure

```
infrastructure/
├── README.md                           ← YOU ARE HERE
│
├── amd_migration/                      ← GPU upgrade scripts
│   ├── 1_pre_migration_checklist.sh   Run BEFORE GPU swap
│   ├── 2_install_rocm.sh               Install AMD drivers
│   ├── 3_install_pytorch_rocm.sh       Install PyTorch ROCm
│   ├── 4_verify_amd_setup.sh           Test GPU is working
│   └── 5_update_training_config.sh     Configure for 10 workers
│
├── tiburtina_integration/              ← Sentiment analysis
│   ├── 1_setup_tiburtina.sh            Install Ollama + Mistral
│   ├── 2_integrate_paper_trader.sh     Add sentiment to trading
│   └── sentiment_integration.patch     Integration guide
│
└── local_coding_bot/                   ← AI coding assistant
    ├── 1_setup_options.md              Compare options
    └── 2_install_coding_bot.sh         Install Aider + Continue
```

---

## 🚀 Quick Start Commands

### **Run Everything (Full Setup)**

```bash
cd /opt/user-data/experiment/cappuccino

# Day 1: Backup
./infrastructure/amd_migration/1_pre_migration_checklist.sh

# Days 2-4: AI Setup
./infrastructure/tiburtina_integration/1_setup_tiburtina.sh
./infrastructure/local_coding_bot/2_install_coding_bot.sh

# Day 5: GPU Swap
# (Physical swap)
./infrastructure/amd_migration/2_install_rocm.sh
# (Reboot)
./infrastructure/amd_migration/3_install_pytorch_rocm.sh
./infrastructure/amd_migration/4_verify_amd_setup.sh
./infrastructure/amd_migration/5_update_training_config.sh

# Start training!
./start_training_amd.sh
```

### **Monitor Everything**

```bash
# AMD GPU
./monitor_amd.sh

# Training progress
tail -f logs/worker_1.log

# Paper trading
tail -f logs/paper_trader_trial250_500usd.log

# Sentiment analysis
python3 test_sentiment_paper_trader.py
```

---

## 🎯 Key Features After Setup

### 1. **High-Performance Training (RX 7900 GRE)**
- 10 parallel workers
- 60-80% trial success rate
- 4-5x faster model discovery
- Larger batch sizes (98K vs 32K)

### 2. **Intelligent Trading (Tiburtina)**
- Real-time sentiment analysis
- LLM-powered news interpretation
- Automatic position sizing adjustment
- Bearish news detection → skip/reduce trades

### 3. **AI-Powered Development (Coding Bot)**
- **Continue**: Tab autocomplete in VSCode
- **Aider**: Autonomous multi-file editing
- **Mistral**: General coding questions
- **DeepSeek Coder**: Specialized code completion

---

## 📚 Documentation

Each component has detailed docs:

- **AMD Migration**: `./amd_migration/MIGRATION_STEPS.txt` (auto-generated)
- **Tiburtina**: `./.claude/tiburtina_integration.md`
- **Coding Bot**: `./local_coding_bot/1_setup_options.md`

---

## 🧪 Testing Commands

```bash
# Test AMD GPU
./infrastructure/amd_migration/4_verify_amd_setup.sh

# Test Tiburtina sentiment
python3 test_sentiment_paper_trader.py

# Test coding bot
./test_ollama.sh
aider --model ollama/mistral

# Test full system
./monitor_amd.sh  # Watch training
tail -f logs/paper_trader_trial250_500usd.log  # Watch trading
```

---

## 💰 Cost Summary

| Component | Cost |
|-----------|------|
| RX 7900 GRE | $300 (after selling RTX 3070) |
| ROCm | $0 (open source) |
| Ollama/Mistral | $0 (local) |
| Tiburtina | $0 (local LLM) |
| Aider | $0 (open source) |
| Continue | $0 (open source) |
| **Total** | **$300 one-time** |

**Ongoing costs:** $0/month (everything runs locally)

---

## 🎁 Bonus: After GPU Upgrade

With 16GB VRAM, you can also run:

```bash
# Bigger, smarter coding models
ollama pull codellama:33b       # 33B parameters
ollama pull deepseek-coder:33b  # State-of-the-art coder
ollama pull mixtral:8x7b        # 47B parameters

# Use in Continue/Aider
# Much better code quality
```

---

## ❓ Troubleshooting

**Problem:** ROCm installation fails
```bash
# Check kernel version
uname -r  # Must be 5.15+

# Check GPU detected
lspci | grep -i amd

# Try Arch wiki
# https://wiki.archlinux.org/title/AMDGPU
```

**Problem:** Ollama models slow
```bash
# Check if using GPU
ollama run mistral "test" --verbose

# Should show GPU usage
# If not, reinstall ROCm first
```

**Problem:** Training OOM on AMD
```bash
# Reduce workers in .env.training
TRAINING_WORKERS=8  # Start with 8 instead of 10

# Or reduce batch size in 1_optimize_unified.py
```

---

## 📞 Support

All scripts include verbose error messages. If stuck:

1. Check the generated logs in `backups/pre_amd_migration/`
2. Run verification scripts to diagnose
3. Each component can be rolled back independently

---

## ✅ Completion Checklist

**Pre-GPU:**
- [ ] Run pre-migration checklist
- [ ] Install Tiburtina
- [ ] Install coding bot
- [ ] Test sentiment analysis
- [ ] Backup Trial #250

**GPU Day:**
- [ ] Physical swap
- [ ] Install ROCm
- [ ] Install PyTorch ROCm
- [ ] Verify GPU working
- [ ] Update training config
- [ ] Start 10 workers
- [ ] Restart paper trader

**Post-GPU:**
- [ ] Monitor for 24 hours
- [ ] Verify success rate >60%
- [ ] Check paper trading still running
- [ ] Test Tiburtina sentiment in trading
- [ ] Use Aider for first code task

---

**Status: Infrastructure Ready for 5-Day Implementation** 🚀

Questions? Each subdirectory has detailed scripts with help messages.
