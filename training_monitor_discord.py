#!/usr/bin/env python3
"""
Training Monitor with Discord Notifications

Periodically checks training progress and sends updates to Discord.
Runs alongside training workers without interfering.
"""

import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

# Load environment and Discord
from dotenv import load_dotenv
load_dotenv()

from integrations.discord_notifier import DiscordNotifier
from constants import DISCORD

def monitor_and_notify(db_path='databases/ensemble_ft_campaign.db', interval_minutes=30):
    """Monitor training and send Discord updates."""

    if not DISCORD.ENABLED or not DISCORD.NOTIFY_TRAINING:
        print("❌ Discord training notifications disabled in config")
        return

    notifier = DiscordNotifier()
    if not notifier.enabled:
        print("❌ Discord notifier not enabled")
        return

    print("🔔 Training Monitor Started")
    print(f"   Database: {db_path}")
    print(f"   Update interval: {interval_minutes} minutes")
    print(f"   Discord notifications: ENABLED")
    print()

    last_reported = {}  # Track last reported trial number per study

    while True:
        try:
            if not Path(db_path).exists():
                print(f"⏳ Waiting for database: {db_path}")
                time.sleep(60)
                continue

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get all studies
            cursor.execute("SELECT study_id, study_name FROM studies")
            studies = cursor.fetchall()

            if not studies:
                print("⏳ No studies found yet, waiting...")
                time.sleep(60)
                conn.close()
                continue

            # Check each study for new completed trials
            for study_id, study_name in studies:
                # Get statistics
                cursor.execute("""
                    SELECT COUNT(*),
                           SUM(CASE WHEN state = 'COMPLETE' THEN 1 ELSE 0 END)
                    FROM trials
                    WHERE study_id = ?
                """, (study_id,))

                total, complete = cursor.fetchone()

                # Get best trial
                cursor.execute("""
                    SELECT t.trial_id, tv.value, t.number
                    FROM trials t
                    JOIN trial_values tv ON t.trial_id = tv.trial_id
                    WHERE t.study_id = ? AND t.state = 'COMPLETE'
                    ORDER BY tv.value DESC
                    LIMIT 1
                """, (study_id,))

                best = cursor.fetchone()

                if best:
                    best_trial_num, best_sharpe, _ = best

                    # Check if we should send update
                    last_num = last_reported.get(study_name, -1)

                    # Send update every 10 trials or first trial
                    if complete > 0 and (complete % 10 == 0 or complete == 1) and complete != last_num:
                        # Determine study type
                        study_type = "🤖 FT" if "ft_transformer" in study_name.lower() else "📊 Ensemble"

                        # Progress
                        progress_pct = (complete / 100.0 * 100) if complete > 0 else 0

                        # Send notification
                        notifier.send_message(
                            content='',
                            embed={
                                'title': f'{study_type} Training Update',
                                'description': f'{study_name}',
                                'color': 0x00ff00 if best_sharpe > 0 else 0xff9900,
                                'fields': [
                                    {
                                        'name': '🎯 Progress',
                                        'value': f'{complete}/100 trials ({progress_pct:.1f}%)',
                                        'inline': False
                                    },
                                    {
                                        'name': '🏆 Best Sharpe',
                                        'value': f'{best_sharpe:.4f}',
                                        'inline': True
                                    },
                                    {
                                        'name': '📈 Best Trial',
                                        'value': f'Trial #{best_trial_num}',
                                        'inline': True
                                    }
                                ],
                                'timestamp': datetime.now().isoformat(),
                                'footer': {'text': 'Cappuccino Training Monitor'}
                            }
                        )

                        last_reported[study_name] = complete
                        print(f"✅ Sent update for {study_name}: {complete}/100 trials, Best Sharpe: {best_sharpe:.4f}")

            conn.close()

            # Wait before next check
            print(f"⏳ Next check in {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            print("\n👋 Training monitor stopped")
            break
        except Exception as e:
            print(f"❌ Error in monitor: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(60)

if __name__ == '__main__':
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'databases/ensemble_ft_campaign.db'
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    monitor_and_notify(db_path, interval)
