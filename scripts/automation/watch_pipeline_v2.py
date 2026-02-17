#!/usr/bin/env python3
"""
Pipeline V2 Watcher - Auto-refreshing trial progress viewer
"""
import os
import time
import sys
from pipeline_v2_viewer import main as show_status

def watch_pipeline(interval=10):
    """Watch pipeline status with auto-refresh."""
    try:
        while True:
            # Clear screen
            os.system('clear' if os.name != 'nt' else 'cls')

            # Show status
            show_status()

            # Show refresh message
            print(f"\nPress Ctrl+C to stop watching")
            print(f"Refreshing in {interval} seconds...")

            # Sleep
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nStopped watching pipeline.")
        sys.exit(0)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Watch Pipeline V2 status')
    parser.add_argument('--interval', type=int, default=10, help='Refresh interval in seconds')
    args = parser.parse_args()

    watch_pipeline(interval=args.interval)
