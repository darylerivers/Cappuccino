#!/usr/bin/env python3
"""
VIN-Like Trial Naming System

Encodes trial hyperparameters and performance grade into a compact, human-readable identifier.

Format: [TYPE]-[GRADE]-[ARCH]-[OPT]-[ENV]-[TIMESTAMP]

Example: PPO-A-N1024-L2E6-G98-20260214
- PPO: Model type
- A: Performance grade (S=Elite, A=Excellent, B=Good, C=Fair, D=Poor, F=Failed)
- N1024: Network dimension
- L2E6: Learning rate (2e-6)
- G98: Gamma (0.98)
- 20260214: Training date

Full breakdown:
    [TYPE]  = Model type (PPO, DDPG, A2C, etc.)
    [GRADE] = Performance grade based on Sharpe ratio:
              S = Sharpe >= 0.30 (Elite - top 1%)
              A = Sharpe >= 0.20 (Excellent - top 10%)
              B = Sharpe >= 0.15 (Good - top 25%)
              C = Sharpe >= 0.10 (Fair - top 50%)
              D = Sharpe >= 0.05 (Poor - bottom 50%)
              F = Sharpe <  0.05 (Failed)
    [ARCH]  = Architecture: N{dim}-B{batch} (e.g., N1024-B4096)
    [OPT]   = Optimizer: L{lr}-G{gamma} (e.g., L2E6-G98)
    [ENV]   = Environment: LB{lookback}-TD{time_decay} (e.g., LB5-TD20)
    [TS]    = Timestamp: YYYYMMDD
"""

import hashlib
from datetime import datetime
from typing import Dict, Tuple


def sharpe_to_grade(sharpe: float) -> str:
    """Convert Sharpe ratio to letter grade."""
    if sharpe >= 0.30:
        return 'S'  # Elite
    elif sharpe >= 0.20:
        return 'A'  # Excellent (top 10%)
    elif sharpe >= 0.15:
        return 'B'  # Good
    elif sharpe >= 0.10:
        return 'C'  # Fair
    elif sharpe >= 0.05:
        return 'D'  # Poor
    else:
        return 'F'  # Failed


def grade_to_numeric(grade: str) -> int:
    """Convert letter grade to numeric for sorting."""
    grades = {'S': 6, 'A': 5, 'B': 4, 'C': 3, 'D': 2, 'F': 1}
    return grades.get(grade, 0)


def format_learning_rate(lr: float) -> str:
    """Format learning rate compactly (e.g., 2e-6 -> L2E6)."""
    # Convert to scientific notation
    exp = 0
    value = lr
    while value < 1 and exp < 20:
        value *= 10
        exp += 1

    # Round to 1 decimal place
    value = round(value, 1)
    if value == int(value):
        value = int(value)

    return f"L{value}E{exp}"


def format_gamma(gamma: float) -> str:
    """Format gamma compactly (e.g., 0.98 -> G98)."""
    return f"G{int(gamma * 100)}"


def format_net_dim(net_dim: int) -> str:
    """Format network dimension (e.g., 1024 -> N1024)."""
    return f"N{net_dim}"


def format_batch(batch: int) -> str:
    """Format batch size (e.g., 4096 -> B4096, or 4K)."""
    if batch >= 1024:
        return f"B{batch // 1024}K"
    return f"B{batch}"


def format_lookback(lookback: int) -> str:
    """Format lookback period."""
    return f"LB{lookback}"


def format_time_decay(td_floor: float) -> str:
    """Format time decay floor (e.g., 0.20 -> TD20)."""
    return f"TD{int(td_floor * 100)}"


def generate_trial_vin(
    model_type: str,
    sharpe: float,
    hyperparams: Dict,
    timestamp: datetime = None
) -> Tuple[str, str, Dict]:
    """
    Generate a VIN-like identifier for a trial.

    Args:
        model_type: Model type (e.g., 'ppo', 'ddpg')
        sharpe: Sharpe ratio achieved
        hyperparams: Dictionary of hyperparameters
        timestamp: Training timestamp (defaults to now)

    Returns:
        Tuple of (vin_code, grade, metadata_dict)

    Example:
        >>> vin, grade, meta = generate_trial_vin('ppo', 0.25, {
        ...     'net_dimension': 1024,
        ...     'learning_rate': 2e-6,
        ...     'gamma': 0.98,
        ...     'batch_size': 4096,
        ...     'lookback': 5,
        ...     'time_decay_floor': 0.20
        ... })
        >>> print(vin)
        PPO-A-N1024B4K-L2E6G98-LB5TD20-20260214
    """
    if timestamp is None:
        timestamp = datetime.now()

    # Extract parameters
    model = model_type.upper()
    grade = sharpe_to_grade(sharpe)

    # Architecture component
    net_dim = format_net_dim(hyperparams.get('net_dimension', 0))
    batch = format_batch(hyperparams.get('batch_size', 0))
    arch = f"{net_dim}{batch}"

    # Optimizer component
    lr = format_learning_rate(hyperparams.get('learning_rate', 0))
    gamma = format_gamma(hyperparams.get('gamma', 0))
    opt = f"{lr}{gamma}"

    # Environment component
    lookback = format_lookback(hyperparams.get('lookback', 0))
    time_decay = format_time_decay(hyperparams.get('time_decay_floor', 0))
    env = f"{lookback}{time_decay}"

    # Timestamp
    ts = timestamp.strftime("%Y%m%d")

    # Construct VIN
    vin = f"{model}-{grade}-{arch}-{opt}-{env}-{ts}"

    # Metadata for easy lookup
    metadata = {
        'vin': vin,
        'model_type': model_type,
        'grade': grade,
        'grade_numeric': grade_to_numeric(grade),
        'sharpe': sharpe,
        'timestamp': timestamp.isoformat(),
        'hyperparams': hyperparams,
        'components': {
            'model': model,
            'grade': grade,
            'architecture': arch,
            'optimizer': opt,
            'environment': env,
            'date': ts
        }
    }

    return vin, grade, metadata


def generate_trial_hash(vin: str) -> str:
    """Generate a short hash from VIN for filesystem-safe names."""
    return hashlib.md5(vin.encode()).hexdigest()[:8]


def parse_vin(vin: str) -> Dict:
    """
    Parse a VIN code back into components.

    Example:
        >>> parse_vin("PPO-A-N1024B4K-L2E6G98-LB5TD20-20260214")
        {'model': 'PPO', 'grade': 'A', ...}
    """
    parts = vin.split('-')
    if len(parts) != 6:
        raise ValueError(f"Invalid VIN format: {vin}")

    return {
        'model': parts[0],
        'grade': parts[1],
        'architecture': parts[2],
        'optimizer': parts[3],
        'environment': parts[4],
        'date': parts[5],
        'full_vin': vin
    }


def is_top_percentile(grade: str, percentile: int = 10) -> bool:
    """
    Check if a grade qualifies for top percentile.

    Args:
        grade: Letter grade (S, A, B, C, D, F)
        percentile: Target percentile (10 = top 10%)

    Returns:
        True if grade qualifies for the percentile
    """
    grade_percentiles = {
        'S': 1,   # Top 1%
        'A': 10,  # Top 10%
        'B': 25,  # Top 25%
        'C': 50,  # Top 50%
        'D': 75,  # Top 75%
        'F': 100  # All
    }

    return grade_percentiles.get(grade, 100) <= percentile


if __name__ == "__main__":
    # Example usage
    example_params = {
        'net_dimension': 1024,
        'learning_rate': 2.5e-6,
        'gamma': 0.98,
        'batch_size': 4096,
        'lookback': 5,
        'time_decay_floor': 0.20,
        'clip_range': 0.20,
        'entropy_coef': 0.002
    }

    for sharpe in [0.35, 0.25, 0.18, 0.12, 0.08, 0.03]:
        vin, grade, metadata = generate_trial_vin('ppo', sharpe, example_params)
        is_top = is_top_percentile(grade, 10)
        print(f"Sharpe {sharpe:.2f} -> {vin} (Top 10%: {is_top})")
