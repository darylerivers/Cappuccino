#!/usr/bin/env python3
"""
Validate Phase 2 Enhanced Data

Checks that rolling mean features were calculated correctly.

Usage:
    python validate_phase2_data.py --data-dir data/1h_1680_phase2
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'


def validate_phase2_data(data_dir):
    """Validate Phase 2 enhanced data."""
    data_path = Path(data_dir)

    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}PHASE 2 DATA VALIDATION{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")

    # Load metadata
    print(f"{Colors.BLUE}Loading metadata...{Colors.END}")
    with open(data_path / 'metadata.json') as f:
        metadata = json.load(f)

    print(f"  Created: {metadata['created']}")
    print(f"  Timesteps: {metadata['timesteps']}")
    print(f"  Base features: {metadata['base_features']}")
    print(f"  Rolling features: {metadata['rolling_features']}")
    print(f"  Total features: {metadata['total_features']}")
    print(f"  State dimension: {metadata['base_state_dim']} → {metadata['enhanced_state_dim']}")

    # Load arrays
    print(f"\n{Colors.BLUE}Loading arrays...{Colors.END}")
    tech_base = np.load(data_path / 'tech_array_base.npy')
    tech_enhanced = np.load(data_path / 'tech_array.npy')
    rolling_features = np.load(data_path / 'rolling_features.npy')

    print(f"  Base tech array: {tech_base.shape}")
    print(f"  Enhanced tech array: {tech_enhanced.shape}")
    print(f"  Rolling features: {rolling_features.shape}")

    # Validation checks
    print(f"\n{Colors.CYAN}Running validation checks...{Colors.END}\n")

    errors = []
    warnings = []

    # Check 1: Shape consistency
    print(f"  [1] Checking shape consistency...", end=' ')
    if tech_enhanced.shape[1] == tech_base.shape[1] + rolling_features.shape[1]:
        print(f"{Colors.GREEN}✓{Colors.END}")
    else:
        print(f"{Colors.RED}✗{Colors.END}")
        errors.append(f"Shape mismatch: {tech_enhanced.shape[1]} != {tech_base.shape[1]} + {rolling_features.shape[1]}")

    # Check 2: Base features unchanged
    print(f"  [2] Checking base features preserved...", end=' ')
    if np.allclose(tech_enhanced[:, :tech_base.shape[1]], tech_base):
        print(f"{Colors.GREEN}✓{Colors.END}")
    else:
        print(f"{Colors.RED}✗{Colors.END}")
        errors.append("Base features were modified")

    # Check 3: Rolling features appended
    print(f"  [3] Checking rolling features appended...", end=' ')
    if np.allclose(tech_enhanced[:, tech_base.shape[1]:], rolling_features):
        print(f"{Colors.GREEN}✓{Colors.END}")
    else:
        print(f"{Colors.RED}✗{Colors.END}")
        errors.append("Rolling features not correctly appended")

    # Check 4: No NaN values
    print(f"  [4] Checking for NaN values...", end=' ')
    nan_count = np.isnan(tech_enhanced).sum()
    if nan_count == 0:
        print(f"{Colors.GREEN}✓{Colors.END}")
    else:
        print(f"{Colors.YELLOW}!{Colors.END}")
        warnings.append(f"Found {nan_count} NaN values in enhanced array")

    # Check 5: No inf values
    print(f"  [5] Checking for inf values...", end=' ')
    inf_count = np.isinf(tech_enhanced).sum()
    if inf_count == 0:
        print(f"{Colors.GREEN}✓{Colors.END}")
    else:
        print(f"{Colors.YELLOW}!{Colors.END}")
        warnings.append(f"Found {inf_count} inf values in enhanced array")

    # Check 6: Rolling means are reasonable
    print(f"  [6] Checking rolling mean values...", end=' ')
    # 7-day close MA should be similar to 30-day close MA in magnitude
    ma7d_first_crypto = rolling_features[:, 0]
    ma30d_first_crypto = rolling_features[:, 2]

    ma7d_mean = np.mean(ma7d_first_crypto)
    ma30d_mean = np.mean(ma30d_first_crypto)

    # They should be within 20% of each other
    if abs(ma7d_mean - ma30d_mean) / max(ma7d_mean, ma30d_mean) < 0.2:
        print(f"{Colors.GREEN}✓{Colors.END}")
    else:
        print(f"{Colors.YELLOW}!{Colors.END}")
        warnings.append(f"7d MA ({ma7d_mean:.2f}) and 30d MA ({ma30d_mean:.2f}) differ significantly")

    # Check 7: Verify manual calculation for one crypto
    print(f"  [7] Verifying manual calculation...", end=' ')
    # Extract close prices for first crypto from base tech array
    # Tech array structure: crypto_0 [open, high, low, close, volume, ...] = 11 features
    close_idx = 3  # close is 4th indicator
    close_prices = tech_base[:, close_idx]

    # Calculate 7-day MA manually
    window_7d = metadata['rolling_window_short'] * 24  # 7 days * 24 hours
    close_series = pd.Series(close_prices)
    manual_ma7d = close_series.rolling(window=window_7d, min_periods=1).mean()

    # Compare with stored values
    if np.allclose(rolling_features[:, 0], manual_ma7d.values, rtol=1e-5):
        print(f"{Colors.GREEN}✓{Colors.END}")
    else:
        print(f"{Colors.RED}✗{Colors.END}")
        errors.append("Manual calculation doesn't match stored rolling features")

    # Summary
    print(f"\n{Colors.CYAN}Validation Summary:{Colors.END}")
    print(f"  Checks passed: {7 - len(errors) - len(warnings)}/7")
    print(f"  Errors: {len(errors)}")
    print(f"  Warnings: {len(warnings)}")

    if errors:
        print(f"\n{Colors.RED}Errors:{Colors.END}")
        for err in errors:
            print(f"  • {err}")

    if warnings:
        print(f"\n{Colors.YELLOW}Warnings:{Colors.END}")
        for warn in warnings:
            print(f"  • {warn}")

    # Feature statistics
    print(f"\n{Colors.CYAN}Feature Statistics (First Crypto):{Colors.END}")
    print(f"  7-day close MA:")
    print(f"    Mean: {ma7d_first_crypto.mean():.2f}")
    print(f"    Std:  {ma7d_first_crypto.std():.2f}")
    print(f"    Min:  {ma7d_first_crypto.min():.2f}")
    print(f"    Max:  {ma7d_first_crypto.max():.2f}")

    print(f"  30-day close MA:")
    print(f"    Mean: {ma30d_first_crypto.mean():.2f}")
    print(f"    Std:  {ma30d_first_crypto.std():.2f}")
    print(f"    Min:  {ma30d_first_crypto.min():.2f}")
    print(f"    Max:  {ma30d_first_crypto.max():.2f}")

    # Final result
    print(f"\n{Colors.HEADER}{'='*80}{Colors.END}")
    if not errors:
        print(f"{Colors.GREEN}✓ VALIDATION PASSED!{Colors.END}")
        print(f"\nPhase 2 data is ready for training.")
        result = 0
    else:
        print(f"{Colors.RED}✗ VALIDATION FAILED!{Colors.END}")
        print(f"\nPlease fix errors before using this data.")
        result = 1

    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate Phase 2 Enhanced Data")
    parser.add_argument('--data-dir', type=str, default='data/1h_1680_phase2',
                       help='Phase 2 data directory')
    args = parser.parse_args()

    return validate_phase2_data(args.data_dir)


if __name__ == '__main__':
    exit(main())
