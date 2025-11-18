"""
Trial #13 - PRODUCTION BEST CONFIGURATION
==========================================
Objective Value (Sharpe Δ): 0.077980
Date: 2025-10-29
Study: cappuccino_exploitation

This configuration achieved 33% better performance than the initial best trial.
156 subsequent trials could not beat it, indicating strong convergence to optimal.
"""

# ============================================================================
# BEST HYPERPARAMETERS - TRIAL #13
# ============================================================================

BEST_CONFIG = {
    # Core Learning Parameters
    "learning_rate": 1.3314e-06,  # Very low, stable learning
    "batch_size": 3072,  # Large batches for stable gradients
    "gamma": 0.98,  # High discount factor (values long-term)
    "net_dimension": 1536,  # Network size

    # Training Control
    "target_step": 290,  # Steps per update
    "break_step": 130000,  # Total training steps
    "worker_num": 11,  # Parallel environments
    "thread_num": 12,  # CPU threads

    # PPO-Specific
    "clip_range": 0.15,  # Conservative policy updates
    "entropy_coef": 0.0034017398,  # Exploration bonus
    "value_loss_coef": 0.95,  # Strong value function emphasis
    "max_grad_norm": 1.1,  # Gradient clipping
    "gae_lambda": 0.95,  # GAE parameter

    # Learning Rate Schedule
    "use_lr_schedule": True,
    "lr_schedule_type": "linear",
    "lr_schedule_factor": 0.85,

    # Environment Normalization
    "lookback": 5,
    "norm_cash_exp": -15,
    "norm_stocks_exp": -9,
    "norm_tech_exp": -17,
    "norm_reward_exp": -11,
    "norm_action": 20000,

    # Risk Management
    "min_cash_reserve": 0.05,
    "concentration_penalty": 0.05,
    "time_decay_floor": 0.2,

    # Evaluation
    "eval_time_gap": 60,
}

# Derived parameters
BEST_CONFIG["use_multiprocessing"] = False

if __name__ == "__main__":
    print("=" * 80)
    print("PRODUCTION CONFIGURATION - TRIAL #13")
    print("Sharpe Delta: 0.077980 (33% better than initial best)")
    print("=" * 80)
    for key, value in BEST_CONFIG.items():
        print(f"{key:25s}: {value}")
