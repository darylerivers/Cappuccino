#!/usr/bin/env python3
"""
Monitor training and alert when an FT-Transformer trial becomes #1.

Usage:
    python monitor_ft_winner.py --study cappuccino_ft_transformer --check-interval 300
"""

import argparse
import sqlite3
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


def get_best_trial(db_path: str, study_name: str):
    """Get the current best trial and check if it's FT-Transformer."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get study_id
        cursor.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
        result = cursor.fetchone()
        if not result:
            return None
        study_id = result[0]

        # Get best trial
        cursor.execute("""
            SELECT t.number, tv.value, t.trial_id
            FROM trials t
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.study_id = ? AND t.state = 'COMPLETE' AND tv.value IS NOT NULL
            ORDER BY tv.value DESC
            LIMIT 1
        """, (study_id,))

        best = cursor.fetchone()
        if not best:
            conn.close()
            return None

        trial_num, value, trial_id = best

        # Check if it's FT-Transformer
        cursor.execute("""
            SELECT param_value
            FROM trial_params
            WHERE trial_id = ? AND param_name = 'use_ft_encoder'
        """, (trial_id,))

        ft_result = cursor.fetchone()
        is_ft = ft_result and ft_result[0] == 1.0

        # Get FT config if applicable
        ft_config = {}
        if is_ft:
            cursor.execute("""
                SELECT param_name, param_value
                FROM trial_params
                WHERE trial_id = ? AND param_name LIKE 'ft_%'
            """, (trial_id,))
            for name, value in cursor.fetchall():
                ft_config[name] = value

        conn.close()

        return {
            'number': trial_num,
            'value': value,
            'is_ft': is_ft,
            'ft_config': ft_config
        }

    except Exception as e:
        print(f"Error querying database: {e}")
        return None


def get_trial_counts(db_path: str, study_name: str):
    """Get total trial counts."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get study_id
        cursor.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
        result = cursor.fetchone()
        if not result:
            conn.close()
            return None
        study_id = result[0]

        # Count trials
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN t.state = 'COMPLETE' THEN 1 ELSE 0 END) as complete
            FROM trials t
            WHERE t.study_id = ?
        """, (study_id,))

        total, complete = cursor.fetchone()

        # Count FT trials
        cursor.execute("""
            SELECT COUNT(DISTINCT t.trial_id)
            FROM trials t
            JOIN trial_params tp ON t.trial_id = tp.trial_id
            WHERE t.study_id = ? AND tp.param_name = 'use_ft_encoder' AND tp.param_value = 1.0
        """, (study_id,))
        ft_count = cursor.fetchone()[0]

        conn.close()

        return {
            'total': total or 0,
            'complete': complete or 0,
            'ft_count': ft_count,
            'baseline_count': (total or 0) - ft_count
        }

    except Exception as e:
        print(f"Error querying database: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Monitor for FT-Transformer trial winning')
    parser.add_argument('--study', type=str, default='cappuccino_ft_transformer',
                        help='Optuna study name')
    parser.add_argument('--db', type=str, default='databases/optuna_cappuccino.db',
                        help='Optuna database path')
    parser.add_argument('--check-interval', type=int, default=300,
                        help='Seconds between checks (default: 300 = 5 min)')
    parser.add_argument('--alert-sound', action='store_true',
                        help='Play sound when FT trial wins (requires beep)')

    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"{Colors.RED}Error: Database not found: {db_path}{Colors.END}")
        return

    print(f"{Colors.HEADER}{'='*80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}FT-TRANSFORMER WINNER MONITOR{Colors.END}")
    print(f"{Colors.HEADER}{'='*80}{Colors.END}\n")
    print(f"Study: {args.study}")
    print(f"Database: {db_path}")
    print(f"Check interval: {args.check_interval}s ({args.check_interval/60:.1f} minutes)")
    print(f"\n{Colors.CYAN}Monitoring for FT-Transformer trial to take #1 spot...{Colors.END}")
    print(f"{Colors.YELLOW}Press Ctrl+C to stop{Colors.END}\n")

    last_best = None
    check_count = 0

    try:
        while True:
            check_count += 1
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Get current best trial
            best = get_best_trial(db_path, args.study)
            counts = get_trial_counts(db_path, args.study)

            if not best or not counts:
                print(f"[{now}] No data available yet... (check #{check_count})")
                time.sleep(args.check_interval)
                continue

            # Check if best trial changed
            current_best_num = best['number']
            is_new_leader = last_best is None or current_best_num != last_best

            if is_new_leader:
                print(f"\n{Colors.CYAN}{'='*80}{Colors.END}")
                print(f"{Colors.CYAN}[{now}] NEW LEADER DETECTED (Check #{check_count}){Colors.END}")
                print(f"{Colors.CYAN}{'='*80}{Colors.END}")
                print(f"Trial #{best['number']} - Sharpe: {best['value']:.6f}")
                print(f"Type: {Colors.CYAN}FT-Transformer{Colors.END}" if best['is_ft'] else f"Type: Baseline MLP")

                if best['is_ft']:
                    print(f"\n{Colors.GREEN}{'='*80}{Colors.END}")
                    print(f"{Colors.GREEN}{Colors.BOLD}🎉 FT-TRANSFORMER TRIAL IS NOW #1! 🎉{Colors.END}")
                    print(f"{Colors.GREEN}{'='*80}{Colors.END}\n")

                    print(f"Trial: #{best['number']}")
                    print(f"Sharpe Ratio: {Colors.GREEN}{best['value']:.6f}{Colors.END}")
                    print(f"Total Trials: {counts['total']} ({counts['complete']} complete)")
                    print(f"FT Trials: {counts['ft_count']} ({counts['ft_count']/max(counts['total'],1)*100:.1f}%)")

                    if best['ft_config']:
                        print(f"\nFT Configuration:")
                        for key, value in sorted(best['ft_config'].items()):
                            print(f"  {key}: {value}")

                    print(f"\n{Colors.GREEN}{'='*80}{Colors.END}")
                    print(f"{Colors.GREEN}READY FOR PHASE 7: PAPER TRADING DEPLOYMENT{Colors.END}")
                    print(f"{Colors.GREEN}{'='*80}{Colors.END}\n")

                    # Play alert sound if requested
                    if args.alert_sound:
                        try:
                            import os
                            os.system('beep -f 1000 -l 200 -n -f 1200 -l 200 -n -f 1500 -l 400')
                        except:
                            pass

                    print(f"Next steps:")
                    print(f"  1. Review trial details in Optuna dashboard")
                    print(f"  2. Deploy to paper trading:")
                    print(f"     python monitor_training_dashboard.py --study {args.study}")
                    print(f"  3. Wait for validation to save best trial")
                    print(f"  4. Run paper trader with FT model\n")

                    print(f"{Colors.YELLOW}Monitor will continue checking for updates...{Colors.END}\n")

                else:
                    print(f"  (Baseline still leading)")

                last_best = current_best_num

            else:
                # Same leader, just show status
                leader_type = "FT-Trans" if best['is_ft'] else "Baseline"
                print(f"[{now}] Check #{check_count}: Trial #{best['number']} ({leader_type}) still #1 "
                      f"- Sharpe {best['value']:.6f} - {counts['complete']} trials complete")

            time.sleep(args.check_interval)

    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Monitoring stopped by user{Colors.END}")
        print(f"Total checks: {check_count}\n")


if __name__ == '__main__':
    main()
