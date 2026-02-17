#!/usr/bin/env python3
"""
Paper Trading Aggregator
Consolidates individual paper trading CSVs into efficient storage for analysis.

Features:
- Combines all trial sessions into single database
- Preserves trial metadata (model, Sharpe, parameters)
- Supports SQLite and Parquet output
- Archives processed CSVs to keep paper_trades/ clean
"""

import argparse
import json
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


class PaperTradingAggregator:
    def __init__(
        self,
        paper_trades_dir: str = "paper_trades",
        archive_dir: str = "paper_trades/archive",
        output_db: str = "paper_trades/aggregated.db",
        output_parquet: str = "paper_trades/aggregated.parquet",
    ):
        self.paper_trades_dir = Path(paper_trades_dir)
        self.archive_dir = Path(archive_dir)
        self.output_db = output_db
        self.output_parquet = output_parquet

        self.archive_dir.mkdir(exist_ok=True, parents=True)

    def get_trial_metadata(self, trial_num: int) -> Dict:
        """Get trial metadata from Optuna database."""
        try:
            import optuna
            study = optuna.load_study(
                study_name="cappuccino_5m_fresh",
                storage="sqlite:///databases/optuna_cappuccino.db"
            )

            for trial in study.get_trials(deepcopy=False):
                if trial.number == trial_num:
                    return {
                        "trial_number": trial_num,
                        "sharpe": trial.value if trial.value else 0.0,
                        "state": trial.state.name,
                        "datetime_complete": str(trial.datetime_complete) if trial.datetime_complete else None,
                        "params": trial.params,
                    }
        except Exception as e:
            print(f"Warning: Could not load metadata for trial {trial_num}: {e}")

        return {"trial_number": trial_num}

    def extract_trial_number(self, filename: str) -> int:
        """Extract trial number from filename."""
        # trial15_session.csv -> 15
        # trial123_session.csv -> 123
        import re
        match = re.search(r'trial(\d+)', filename)
        if match:
            return int(match.group(1))
        return -1

    def process_csv(self, csv_path: Path) -> pd.DataFrame:
        """Load and enrich CSV with metadata."""
        try:
            df = pd.read_csv(csv_path)

            if len(df) == 0:
                print(f"  Skipping empty: {csv_path.name}")
                return None

            # Add metadata columns
            trial_num = self.extract_trial_number(csv_path.name)
            metadata = self.get_trial_metadata(trial_num)

            df['trial_number'] = trial_num
            df['trial_sharpe'] = metadata.get('sharpe', 0.0)
            df['session_file'] = csv_path.name
            df['archived_at'] = datetime.now().isoformat()

            # Add parameters as JSON
            df['trial_params'] = json.dumps(metadata.get('params', {}))

            return df

        except Exception as e:
            print(f"  Error processing {csv_path.name}: {e}")
            return None

    def aggregate_to_sqlite(self, dfs: List[pd.DataFrame]):
        """Aggregate dataframes into SQLite database."""
        if not dfs:
            print("No data to aggregate")
            return

        combined = pd.concat(dfs, ignore_index=True)

        # Create/append to database
        conn = sqlite3.connect(self.output_db)

        # Write data
        combined.to_sql('paper_trading_sessions', conn, if_exists='append', index=False)

        # Create indexes for efficient queries
        cursor = conn.cursor()
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_trial_number ON paper_trading_sessions(trial_number)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON paper_trading_sessions(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_file ON paper_trading_sessions(session_file)
        """)
        conn.commit()
        conn.close()

        print(f"✅ Aggregated {len(combined)} rows into {self.output_db}")

    def aggregate_to_parquet(self, dfs: List[pd.DataFrame]):
        """Aggregate dataframes into Parquet file."""
        if not dfs:
            return

        combined = pd.concat(dfs, ignore_index=True)

        # Append or create parquet
        parquet_path = Path(self.output_parquet)
        if parquet_path.exists():
            existing = pd.read_parquet(parquet_path)
            combined = pd.concat([existing, combined], ignore_index=True)

        combined.to_parquet(parquet_path, compression='snappy', index=False)
        print(f"✅ Aggregated {len(combined)} rows into {self.output_parquet}")

    def archive_csv(self, csv_path: Path):
        """Move CSV to archive directory."""
        archive_path = self.archive_dir / csv_path.name
        shutil.move(str(csv_path), str(archive_path))

    def run(self, archive_after: bool = True, use_parquet: bool = True):
        """Run aggregation process."""
        print("=" * 80)
        print("PAPER TRADING AGGREGATOR")
        print("=" * 80)

        # Find all session CSVs (exclude current active ones modified in last hour)
        import time
        csv_files = []
        for csv in self.paper_trades_dir.glob("*_session.csv"):
            age_hours = (time.time() - csv.stat().st_mtime) / 3600
            if age_hours > 1:  # Only process files older than 1 hour
                csv_files.append(csv)

        if not csv_files:
            print("No CSV files to aggregate (all are recent/active)")
            return

        print(f"Found {len(csv_files)} CSV files to aggregate")

        # Process CSVs
        dfs = []
        for csv_path in csv_files:
            print(f"Processing: {csv_path.name}")
            df = self.process_csv(csv_path)
            if df is not None:
                dfs.append(df)

        if not dfs:
            print("No valid data to aggregate")
            return

        # Aggregate to SQLite
        self.aggregate_to_sqlite(dfs)

        # Aggregate to Parquet (optional)
        if use_parquet:
            self.aggregate_to_parquet(dfs)

        # Archive original CSVs
        if archive_after:
            print("\nArchiving processed CSVs...")
            for csv_path in csv_files:
                if csv_path.exists():
                    self.archive_csv(csv_path)
                    print(f"  Archived: {csv_path.name}")

        print("\n" + "=" * 80)
        print("AGGREGATION COMPLETE")
        print("=" * 80)
        print(f"SQLite database: {self.output_db}")
        if use_parquet:
            print(f"Parquet file: {self.output_parquet}")
        print(f"Archived CSVs: {self.archive_dir}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate paper trading CSVs")
    parser.add_argument("--no-archive", action="store_true", help="Don't archive CSVs after processing")
    parser.add_argument("--no-parquet", action="store_true", help="Don't create Parquet file")
    args = parser.parse_args()

    aggregator = PaperTradingAggregator()
    aggregator.run(
        archive_after=not args.no_archive,
        use_parquet=not args.no_parquet
    )


if __name__ == "__main__":
    main()
