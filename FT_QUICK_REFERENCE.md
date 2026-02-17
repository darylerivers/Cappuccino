# FT-Transformer Quick Reference Card

**One-page cheat sheet for FT-Transformer usage**

---

## ⚡ Quick Commands

### Test Integration
```bash
# Verify everything works
python test_ft_transformer.py
```

### Pre-Train Encoder (Do This First!)
```bash
# Default settings (recommended)
python scripts/training/pretrain_ft_encoder.py \
    --data-dir data/1h_1680 \
    --epochs 30

# Output: train_results/pretrained_encoders/ft_encoder_TIMESTAMP/best_encoder.pth
```

### Use in Training
```python
# In 1_optimize_unified.py
from drl_agents.agents import AgentPPO_FT

args.use_ft_encoder = True
args.ft_config = {'d_token': 64, 'n_blocks': 2, 'n_heads': 4, 'dropout': 0.1}
args.pretrained_encoder_path = 'train_results/pretrained_encoders/ft_encoder_TIMESTAMP/best_encoder.pth'
args.freeze_encoder = False

agent_class = AgentPPO_FT
```

---

## 🎛️ Hyperparameter Presets

### Small/Fast
```python
'd_token': 32, 'n_blocks': 1, 'n_heads': 2, 'dropout': 0.05
# ~400K params, ~1.2 GB VRAM, 5% slower
```

### Balanced (Default) ⭐
```python
'd_token': 64, 'n_blocks': 2, 'n_heads': 4, 'dropout': 0.1
# ~1.8M params, ~1.8 GB VRAM, 10% slower
```

### Large/Powerful
```python
'd_token': 128, 'n_blocks': 3, 'n_heads': 8, 'dropout': 0.15
# ~7M params, ~3.5 GB VRAM, 20% slower
```

---

## 📊 Worker Scaling (RTX 3060 12GB)

| Config | VRAM/Worker | Max Workers |
|--------|-------------|-------------|
| Baseline (no FT) | ~850 MB | 10 |
| Small FT | ~1.2 GB | 7 |
| Default FT | ~1.8 GB | 5 |
| Large FT | ~3.5 GB | 2 |

---

## 🔍 Troubleshooting

### OOM (Out of Memory)
- Reduce workers: `scripts/automation/training_control.sh` → scale to 2-4 workers
- Use smaller config: `d_token=32, n_blocks=1`
- Run on CPU (slow): `--gpu -1`

### Slow Training
- Freeze encoder: `args.freeze_encoder = True`
- Use smaller config: `d_token=32, n_blocks=1`
- Skip pre-training (not recommended)

### NaN Loss in Pre-training
- Lower learning rate: `--lr 1e-4`
- Increase mask ratio: `--mask-ratio 0.2`
- Check data for NaN: `np.isnan(tech_array).sum()`

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `models/ft_transformer_encoder.py` | Core encoder |
| `drl_agents/agents/net_ft.py` | Enhanced Actor/Critic |
| `drl_agents/agents/AgentPPO_FT.py` | Enhanced PPO agent |
| `scripts/training/pretrain_ft_encoder.py` | Pre-training |
| `docs/FT_TRANSFORMER_INTEGRATION.md` | Full docs |
| `test_ft_transformer.py` | Integration tests |

---

## 🎯 Typical Workflow

```bash
# 1. Pre-train encoder (~15 min)
python scripts/training/pretrain_ft_encoder.py --data-dir data/1h_1680 --epochs 30

# 2. Edit training script
vim scripts/training/1_optimize_unified.py
# Add: from drl_agents.agents import AgentPPO_FT
# Add: args.use_ft_encoder = True, args.ft_config = {...}, args.pretrained_encoder_path = "..."
# Change: agent_class = AgentPPO_FT

# 3. Run single trial (test)
python scripts/training/1_optimize_unified.py --n-trials 1

# 4. Run full Optuna study
python scripts/training/1_optimize_unified.py --n-trials 50

# 5. Check logs
tail -f logs/training_worker_*.log | grep "FT-Transformer"
```

---

## 🧪 Verify It's Working

Look for this in logs:
```
======================================================================
Using FT-Transformer Feature Encoding
======================================================================
Config: {'d_token': 64, 'n_blocks': 2, 'n_heads': 4, 'dropout': 0.1}
Pre-trained encoder: train_results/pretrained_encoders/.../best_encoder.pth
✓ Loaded encoder from epoch 30
  Pre-training val loss: 0.004523
✓ Encoder weights will be fine-tuned during RL training
======================================================================
```

---

## 📞 Get Help

1. Check `docs/FT_TRANSFORMER_INTEGRATION.md` (detailed troubleshooting)
2. Run `python test_ft_transformer.py` (verify installation)
3. Review `IMPLEMENTATION_SUMMARY.md` (architecture overview)
4. Check `FT_TRANSFORMER_STATUS.md` (implementation progress)

---

**Status:** ✅ Phases 0-4 Complete | 🔄 Phases 5-7 Pending
**Last Updated:** February 6, 2026
