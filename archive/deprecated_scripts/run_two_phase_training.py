#!/usr/bin/env python3
"""
Two-Phase Training Orchestrator

Master script that runs both Phase 1 and Phase 2 sequentially:
1. Phase 1: Time-frame optimization (25 combos × 20 trials = 500 trials)
2. Phase 2: Feature-enhanced training with winner (200 PPO + 200 DDQN = 400 trials)

Total: 900 trials across both phases

Features:
- Automatic Phase 1 → Phase 2 progression
- Progress monitoring and checkpoints
- Result aggregation and comparison
- Support for mini-test mode
- Resume capability

Usage:
    # Full run (900 trials)
    python run_two_phase_training.py

    # Mini test (2 combos × 5 trials = 10 for Phase 1, 5 PPO + 5 DDQN = 10 for Phase 2)
    python run_two_phase_training.py --mini-test

    # Resume from checkpoint
    python run_two_phase_training.py --resume two_phase_checkpoint.json

    # Run only Phase 1
    python run_two_phase_training.py --phase1-only

    # Run only Phase 2 (requires Phase 1 winner)
    python run_two_phase_training.py --phase2-only

    # Custom trial counts
    python run_two_phase_training.py --phase1-trials 10 --phase2-ppo-trials 50 --phase2-ddqn-trials 50
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_banner(text, color=Colors.HEADER):
    """Print a formatted banner."""
    width = 100
    print(f"\n{'='*width}")
    print(f"{color}{Colors.BOLD}{text.center(width)}{Colors.END}")
    print(f"{'='*width}\n")


def print_section(text, color=Colors.CYAN):
    """Print a section header."""
    print(f"\n{color}{Colors.BOLD}{text}{Colors.END}")
    print(f"{'-'*len(text)}")


def run_command(cmd, description):
    """
    Run a shell command and capture output.

    Args:
        cmd: Command list for subprocess
        description: Human-readable description

    Returns:
        Tuple of (success, stdout, stderr)
    """
    print(f"\n{Colors.BLUE}▶ {description}{Colors.END}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"\n{Colors.CYAN}{'='*60}")
    print("Live output:")
    print(f"{'='*60}{Colors.END}\n")

    try:
        # Stream output in real-time to terminal
        result = subprocess.run(
            cmd,
            check=True
        )
        print(f"\n{Colors.CYAN}{'='*60}")
        print(f"{Colors.GREEN}✓ Training completed successfully{Colors.END}")
        print(f"{Colors.CYAN}{'='*60}{Colors.END}\n")
        return True, "", ""

    except subprocess.CalledProcessError as e:
        print(f"\n{Colors.RED}✗ Training failed{Colors.END}")
        print(f"{Colors.RED}Exit code: {e.returncode}{Colors.END}")
        return False, "", ""


def check_prerequisites():
    """
    Check if all prerequisites are met.

    Returns:
        Boolean indicating if prerequisites are met
    """
    print_section("Checking Prerequisites", Colors.CYAN)

    checks = []

    # Check Python packages
    required_packages = ['numpy', 'optuna', 'torch']
    for package in required_packages:
        try:
            __import__(package)
            print(f"  {Colors.GREEN}✓{Colors.END} {package} installed")
            checks.append(True)
        except ImportError:
            print(f"  {Colors.RED}✗{Colors.END} {package} not installed")
            checks.append(False)

    # Check required files
    required_files = [
        'config_two_phase.py',
        'phase1_timeframe_optimizer.py',
        'phase2_feature_maximizer.py',
        'timeframe_constraint.py',
        'fee_tier_manager.py',
        'environment_Alpaca_phase2.py',
    ]

    for file in required_files:
        if os.path.exists(file):
            print(f"  {Colors.GREEN}✓{Colors.END} {file} exists")
            checks.append(True)
        else:
            print(f"  {Colors.RED}✗{Colors.END} {file} missing")
            checks.append(False)

    # Check data directory
    if os.path.exists('data'):
        print(f"  {Colors.GREEN}✓{Colors.END} data directory exists")
        checks.append(True)
    else:
        print(f"  {Colors.RED}✗{Colors.END} data directory missing")
        checks.append(False)

    all_passed = all(checks)
    if all_passed:
        print(f"\n{Colors.GREEN}All prerequisites met!{Colors.END}")
    else:
        print(f"\n{Colors.RED}Some prerequisites are missing. Please fix before continuing.{Colors.END}")

    return all_passed


def run_phase1(args):
    """
    Run Phase 1: Time-frame Optimization.

    Args:
        args: Command-line arguments

    Returns:
        Dictionary with Phase 1 results or None if failed
    """
    print_banner("PHASE 1: TIME-FRAME OPTIMIZATION")

    start_time = time.time()

    # Build command
    cmd = [sys.executable, 'phase1_timeframe_optimizer.py']

    if args.mini_test:
        cmd.append('--mini-test')
    if args.phase1_trials:
        cmd.extend(['--trials-per-combo', str(args.phase1_trials)])
    if args.data_dir:
        cmd.extend(['--data-dir', args.data_dir])
    if args.num_paths:
        cmd.extend(['--num-paths', str(args.num_paths)])
    if args.k_test_groups:
        cmd.extend(['--k-test-groups', str(args.k_test_groups)])

    # Run Phase 1
    success, stdout, stderr = run_command(cmd, "Running Phase 1 Optimization")

    if not success:
        print(f"\n{Colors.RED}Phase 1 failed. Check logs above.{Colors.END}")
        return None

    # Load results
    winner_file = 'phase1_winner.json'
    if not os.path.exists(winner_file):
        print(f"\n{Colors.RED}Phase 1 winner file not found: {winner_file}{Colors.END}")
        return None

    with open(winner_file, 'r') as f:
        winner = json.load(f)

    elapsed = time.time() - start_time

    # Print results
    print_section("Phase 1 Results", Colors.GREEN)
    print(f"  Time-frame: {winner['timeframe']}")
    print(f"  Interval: {winner['interval']}")
    print(f"  Best value: {winner['best_value']:.4f}")
    print(f"  Best Sharpe (bot): {winner['best_sharpe_bot']:.4f}")
    print(f"  Best Sharpe (HODL): {winner['best_sharpe_hodl']:.4f}")
    print(f"  Total trials: {winner['n_trials']}")
    print(f"  Duration: {elapsed/3600:.2f} hours")

    winner['phase1_duration'] = elapsed

    return winner


def run_phase2(args, phase1_winner=None):
    """
    Run Phase 2: Feature-Enhanced Training.

    Args:
        args: Command-line arguments
        phase1_winner: Phase 1 winner dictionary (optional, will load from file if None)

    Returns:
        Dictionary with Phase 2 results or None if failed
    """
    print_banner("PHASE 2: FEATURE-ENHANCED TRAINING")

    # Check for Phase 1 winner
    if phase1_winner is None:
        winner_file = 'phase1_winner.json'
        if not os.path.exists(winner_file):
            print(f"{Colors.RED}Phase 1 winner file not found: {winner_file}{Colors.END}")
            print(f"{Colors.YELLOW}Run Phase 1 first or specify --resume{Colors.END}")
            return None

        with open(winner_file, 'r') as f:
            phase1_winner = json.load(f)

    print(f"\n{Colors.CYAN}Using Phase 1 Winner:{Colors.END}")
    print(f"  Time-frame: {phase1_winner['timeframe']}")
    print(f"  Interval: {phase1_winner['interval']}")

    start_time = time.time()

    # Build command
    cmd = [sys.executable, 'phase2_feature_maximizer.py']

    if args.mini_test:
        cmd.append('--mini-test')
    if args.algorithm:
        cmd.extend(['--algorithm', args.algorithm])
    if args.phase2_ppo_trials:
        cmd.extend(['--trials-ppo', str(args.phase2_ppo_trials)])
    if args.phase2_ddqn_trials:
        cmd.extend(['--trials-ddqn', str(args.phase2_ddqn_trials)])
    if args.data_dir:
        cmd.extend(['--data-dir', args.data_dir])
    if args.months:
        cmd.extend(['--months', str(args.months)])
    if args.num_paths:
        cmd.extend(['--num-paths', str(args.num_paths)])
    if args.k_test_groups:
        cmd.extend(['--k-test-groups', str(args.k_test_groups)])

    # Run Phase 2
    success, stdout, stderr = run_command(cmd, "Running Phase 2 Optimization")

    if not success:
        print(f"\n{Colors.RED}Phase 2 failed. Check logs above.{Colors.END}")
        return None

    # Load results
    results = {}
    for alg in ['ppo', 'ddqn']:
        result_file = f'phase2_{alg}_best.json'
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                results[alg] = json.load(f)

    if not results:
        print(f"\n{Colors.RED}No Phase 2 results found{Colors.END}")
        return None

    elapsed = time.time() - start_time

    # Print results
    print_section("Phase 2 Results", Colors.GREEN)
    for alg, result in results.items():
        print(f"\n  {alg.upper()}:")
        print(f"    Best value: {result['best_value']:.4f}")
        print(f"    Best Sharpe (bot): {result['best_sharpe_bot']:.4f}")
        print(f"    Best Sharpe (HODL): {result['best_sharpe_hodl']:.4f}")
        print(f"    Total trials: {result['n_trials']}")

    print(f"\n  Duration: {elapsed/3600:.2f} hours")

    # Determine winner
    if len(results) > 1:
        winner_alg = max(results.keys(), key=lambda k: results[k]['best_value'])
        print(f"\n  {Colors.BOLD}{Colors.GREEN}🏆 Algorithm Winner: {winner_alg.upper()}{Colors.END}")

    results_summary = {
        'algorithms': results,
        'phase2_duration': elapsed,
        'phase1_winner': phase1_winner
    }

    return results_summary


def generate_final_report(phase1_results, phase2_results, total_start_time):
    """
    Generate final comparison report.

    Args:
        phase1_results: Phase 1 results dictionary
        phase2_results: Phase 2 results dictionary
        total_start_time: Start time of entire run
    """
    print_banner("FINAL REPORT: TWO-PHASE TRAINING COMPLETE")

    total_elapsed = time.time() - total_start_time

    # Phase 1 Summary
    print_section("Phase 1: Time-Frame Optimization", Colors.CYAN)
    print(f"  Winner: {phase1_results['timeframe']} @ {phase1_results['interval']}")
    print(f"  Best value: {phase1_results['best_value']:.4f}")
    print(f"  Best Sharpe: {phase1_results['best_sharpe_bot']:.4f}")
    print(f"  Trials: {phase1_results['n_trials']}")
    print(f"  Duration: {phase1_results.get('phase1_duration', 0)/3600:.2f} hours")

    # Phase 2 Summary
    print_section("Phase 2: Feature-Enhanced Training", Colors.CYAN)
    phase2_algs = phase2_results['algorithms']
    for alg, result in phase2_algs.items():
        print(f"\n  {alg.upper()}:")
        print(f"    Best value: {result['best_value']:.4f}")
        print(f"    Best Sharpe: {result['best_sharpe_bot']:.4f}")
        print(f"    Trials: {result['n_trials']}")

    if len(phase2_algs) > 1:
        winner_alg = max(phase2_algs.keys(), key=lambda k: phase2_algs[k]['best_value'])
        winner_value = phase2_algs[winner_alg]['best_value']
        print(f"\n  {Colors.BOLD}{Colors.GREEN}🏆 Algorithm Winner: {winner_alg.upper()} "
              f"(value={winner_value:.4f}){Colors.END}")

    print(f"\n  Duration: {phase2_results.get('phase2_duration', 0)/3600:.2f} hours")

    # Overall Summary
    print_section("Overall Statistics", Colors.CYAN)
    total_trials = phase1_results['n_trials']
    for result in phase2_algs.values():
        total_trials += result['n_trials']

    print(f"  Total trials: {total_trials}")
    print(f"  Total duration: {total_elapsed/3600:.2f} hours")
    print(f"  Avg time per trial: {total_elapsed/total_trials:.1f} seconds")

    # Save final report
    report = {
        'phase1': phase1_results,
        'phase2': phase2_results,
        'total_trials': total_trials,
        'total_duration': total_elapsed,
        'timestamp': datetime.now().isoformat()
    }

    report_file = 'two_phase_training_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{Colors.GREEN}✓ Final report saved: {report_file}{Colors.END}")

    # Print next steps
    print_section("Next Steps", Colors.YELLOW)
    print("  1. Review the best models:")
    if 'ppo' in phase2_algs:
        print(f"     - PPO: train_results/phase2_ppo/phase2_ppo_trial_{phase2_algs['ppo']['best_trial_number']}/")
    if 'ddqn' in phase2_algs:
        print(f"     - DDQN: train_results/phase2_ddqn/phase2_ddqn_trial_{phase2_algs['ddqn']['best_trial_number']}/")

    print("\n  2. Deploy best model to paper trading:")
    print("     python auto_model_deployer.py")

    print("\n  3. Start autonomous trading:")
    print("     ./start_automation.sh")

    print("\n  4. Monitor performance:")
    print("     python dashboard.py")


def save_checkpoint(phase1_results=None, phase2_results=None, current_phase='phase1'):
    """
    Save checkpoint for resume capability.

    Args:
        phase1_results: Phase 1 results (optional)
        phase2_results: Phase 2 results (optional)
        current_phase: Current phase ('phase1' or 'phase2')
    """
    checkpoint = {
        'current_phase': current_phase,
        'phase1_complete': phase1_results is not None,
        'phase2_complete': phase2_results is not None,
        'phase1_results': phase1_results,
        'phase2_results': phase2_results,
        'timestamp': datetime.now().isoformat()
    }

    checkpoint_file = 'two_phase_checkpoint.json'
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    print(f"\n{Colors.CYAN}Checkpoint saved: {checkpoint_file}{Colors.END}")


def load_checkpoint(checkpoint_file):
    """
    Load checkpoint for resume.

    Args:
        checkpoint_file: Path to checkpoint file

    Returns:
        Checkpoint dictionary or None
    """
    if not os.path.exists(checkpoint_file):
        print(f"{Colors.RED}Checkpoint file not found: {checkpoint_file}{Colors.END}")
        return None

    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)

    print(f"{Colors.GREEN}✓ Checkpoint loaded: {checkpoint_file}{Colors.END}")
    print(f"  Phase 1 complete: {checkpoint['phase1_complete']}")
    print(f"  Phase 2 complete: {checkpoint['phase2_complete']}")
    print(f"  Current phase: {checkpoint['current_phase']}")

    return checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Two-Phase Training Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run (900 trials total)
  python run_two_phase_training.py

  # Mini test (20 trials total)
  python run_two_phase_training.py --mini-test

  # Resume from checkpoint
  python run_two_phase_training.py --resume two_phase_checkpoint.json

  # Run only Phase 1
  python run_two_phase_training.py --phase1-only

  # Run only Phase 2 (requires Phase 1 winner)
  python run_two_phase_training.py --phase2-only
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--mini-test', action='store_true',
                            help='Run mini test (Phase 1: 10 trials, Phase 2: 10 trials)')
    mode_group.add_argument('--phase1-only', action='store_true',
                            help='Run only Phase 1')
    mode_group.add_argument('--phase2-only', action='store_true',
                            help='Run only Phase 2 (requires Phase 1 winner)')

    # Resume
    parser.add_argument('--resume', type=str,
                        help='Resume from checkpoint file')
    parser.add_argument('--skip-prerequisites', action='store_true',
                        help='Skip prerequisite checks')

    # Data configuration
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory (default: data)')
    parser.add_argument('--months', type=int, default=12,
                        help='Data months for Phase 2 (default: 12)')

    # Phase 1 configuration
    parser.add_argument('--phase1-trials', type=int,
                        help='Trials per combination in Phase 1 (default: 20)')

    # Phase 2 configuration
    parser.add_argument('--algorithm', type=str, default='both',
                        choices=['ppo', 'ddqn', 'both'],
                        help='Algorithm to run in Phase 2 (default: both)')
    parser.add_argument('--phase2-ppo-trials', type=int,
                        help='PPO trials in Phase 2 (default: 200)')
    parser.add_argument('--phase2-ddqn-trials', type=int,
                        help='DDQN trials in Phase 2 (default: 200)')

    # CPCV configuration
    parser.add_argument('--num-paths', type=int, default=3,
                        help='CPCV paths (default: 3)')
    parser.add_argument('--k-test-groups', type=int, default=2,
                        help='CPCV test groups (default: 2)')

    args = parser.parse_args()

    # Print header
    print_banner("TWO-PHASE TRAINING ORCHESTRATOR", Colors.HEADER)

    # Check prerequisites
    if not args.skip_prerequisites:
        if not check_prerequisites():
            print(f"\n{Colors.RED}Prerequisites check failed. Exiting.{Colors.END}")
            print(f"{Colors.YELLOW}Use --skip-prerequisites to bypass this check.{Colors.END}")
            sys.exit(1)

    # Handle resume
    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        if checkpoint is None:
            sys.exit(1)

    total_start_time = time.time()

    # Determine what to run
    run_phase1_flag = True
    run_phase2_flag = True

    if args.phase1_only:
        run_phase2_flag = False
    elif args.phase2_only:
        run_phase1_flag = False

    if checkpoint:
        if checkpoint['phase1_complete']:
            print(f"\n{Colors.YELLOW}Phase 1 already complete (from checkpoint){Colors.END}")
            run_phase1_flag = False
        if checkpoint['phase2_complete']:
            print(f"\n{Colors.YELLOW}Phase 2 already complete (from checkpoint){Colors.END}")
            run_phase2_flag = False

    # Run phases
    phase1_results = None
    phase2_results = None

    try:
        # Phase 1
        if run_phase1_flag:
            phase1_results = run_phase1(args)
            if phase1_results is None:
                print(f"\n{Colors.RED}Phase 1 failed. Exiting.{Colors.END}")
                sys.exit(1)
            save_checkpoint(phase1_results=phase1_results, current_phase='phase1')
        elif checkpoint and checkpoint['phase1_results']:
            phase1_results = checkpoint['phase1_results']
            print(f"\n{Colors.GREEN}✓ Using Phase 1 results from checkpoint{Colors.END}")

        # Phase 2
        if run_phase2_flag:
            phase2_results = run_phase2(args, phase1_results)
            if phase2_results is None:
                print(f"\n{Colors.RED}Phase 2 failed. Exiting.{Colors.END}")
                sys.exit(1)
            save_checkpoint(phase1_results=phase1_results, phase2_results=phase2_results, current_phase='phase2')
        elif checkpoint and checkpoint['phase2_results']:
            phase2_results = checkpoint['phase2_results']
            print(f"\n{Colors.GREEN}✓ Using Phase 2 results from checkpoint{Colors.END}")

        # Generate final report if both phases complete
        if phase1_results and phase2_results:
            generate_final_report(phase1_results, phase2_results, total_start_time)

        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ Two-phase training orchestrator complete!{Colors.END}\n")

    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user. Checkpoint saved.{Colors.END}")
        save_checkpoint(phase1_results=phase1_results, phase2_results=phase2_results)
        sys.exit(1)


if __name__ == '__main__':
    main()
