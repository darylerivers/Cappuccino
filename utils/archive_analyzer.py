#!/usr/bin/env python3
"""
Archive Analyzer - Mine historical trial data for insights.

Analyzes archived training trials to find:
- Hyperparameter patterns correlated with performance
- Overlooked configurations that performed well
- Parameter interactions and sweet spots
- Recommendations for future training

Usage:
    python archive_analyzer.py --analyze
    python archive_analyzer.py --analyze --model qwen2.5-coder:7b
    python archive_analyzer.py --export-csv  # Export data for manual analysis
"""

import argparse
import json
import logging
import os
import sqlite3
import tarfile
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class ArchiveAnalyzer:
    """Analyzes archived training trials for insights."""

    def __init__(
        self,
        archive_dir: str = "/var/tmp/cappuccino_archive",
        db_path: str = "databases/optuna_cappuccino.db",
        ollama_model: str = "qwen2.5-coder:7b",
        ollama_url: str = "http://localhost:11434",
    ):
        self.archive_dir = Path(archive_dir)
        self.db_path = db_path
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.manifest_path = self.archive_dir / "archive_manifest.json"

        # Parameter definitions for analysis
        self.param_ranges = {
            'learning_rate': {'min': 1e-6, 'max': 1e-2, 'type': 'log'},
            'batch_size': {'min': 64, 'max': 4096, 'type': 'int'},
            'gamma': {'min': 0.9, 'max': 0.999, 'type': 'float'},
            'gae_lambda': {'min': 0.9, 'max': 0.99, 'type': 'float'},
            'entropy_coef': {'min': 1e-4, 'max': 0.1, 'type': 'log'},
            'vf_coef': {'min': 0.1, 'max': 1.0, 'type': 'float'},
            'max_grad_norm': {'min': 0.1, 'max': 1.0, 'type': 'float'},
            'n_steps': {'min': 256, 'max': 4096, 'type': 'int'},
            'n_epochs': {'min': 3, 'max': 30, 'type': 'int'},
            'clip_range': {'min': 0.1, 'max': 0.4, 'type': 'float'},
            'net_arch_depth': {'min': 1, 'max': 4, 'type': 'int'},
            'net_arch_width': {'min': 64, 'max': 512, 'type': 'int'},
            'lookback': {'min': 1, 'max': 10, 'type': 'int'},
        }

    def load_manifest(self) -> Dict:
        """Load archive manifest."""
        if not self.manifest_path.exists():
            logger.warning(f"No archive manifest found at {self.manifest_path}")
            return {"batches": {}}

        with open(self.manifest_path) as f:
            return json.load(f)

    def get_archived_trial_numbers(self) -> List[int]:
        """Get all trial numbers in archives."""
        manifest = self.load_manifest()
        trial_numbers = []

        for batch_info in manifest.get("batches", {}).values():
            trial_numbers.extend(batch_info.get("trial_numbers", []))

        return sorted(trial_numbers)

    def extract_trial_config(self, archive_path: Path, trial_dir_name: str) -> Optional[Dict]:
        """Extract just the config from a trial in an archive."""
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                # Look for config.json in the trial directory
                config_path = f"{trial_dir_name}/config.json"

                for member in tar.getmembers():
                    if member.name == config_path or member.name.endswith("/config.json"):
                        if trial_dir_name in member.name:
                            f = tar.extractfile(member)
                            if f:
                                return json.load(f)

                # Try hyperparameters.json as fallback
                for member in tar.getmembers():
                    if "hyperparameters.json" in member.name and trial_dir_name in member.name:
                        f = tar.extractfile(member)
                        if f:
                            return json.load(f)

        except Exception as e:
            logger.debug(f"Could not extract config from {trial_dir_name}: {e}")

        return None

    def get_trial_data_from_db(self, trial_numbers: List[int] = None, study_name: str = None) -> pd.DataFrame:
        """Get trial data from Optuna database.

        Args:
            trial_numbers: Optional list of trial numbers to filter
            study_name: Optional study name to filter (or pattern with % wildcard)
        """
        try:
            conn = sqlite3.connect(self.db_path)

            # Get all completed trials with their values
            query = """
            SELECT
                t.trial_id,
                t.number as trial_number,
                s.study_name,
                tv.value as objective_value,
                t.datetime_start,
                t.datetime_complete
            FROM trials t
            JOIN studies s ON t.study_id = s.study_id
            LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
            WHERE t.state = 'COMPLETE'
            """

            params = []

            if study_name:
                if '%' in study_name:
                    query += " AND s.study_name LIKE ?"
                else:
                    query += " AND s.study_name = ?"
                params.append(study_name)

            if trial_numbers:
                placeholders = ','.join('?' * len(trial_numbers))
                query += f" AND t.number IN ({placeholders})"
                params.extend(trial_numbers)

            df_trials = pd.read_sql_query(query, conn, params=params if params else None)

            # Get parameters for each trial
            query_params = """
            SELECT
                t.trial_id,
                tp.param_name,
                tp.param_value
            FROM trial_params tp
            JOIN trials t ON tp.trial_id = t.trial_id
            WHERE t.state = 'COMPLETE'
            """
            df_params = pd.read_sql_query(query_params, conn)

            conn.close()

            # Pivot parameters
            if not df_params.empty:
                df_params_wide = df_params.pivot(
                    index='trial_id',
                    columns='param_name',
                    values='param_value'
                ).reset_index()

                # Merge with trials
                df = df_trials.merge(df_params_wide, on='trial_id', how='left')
            else:
                df = df_trials

            return df

        except Exception as e:
            logger.error(f"Error loading trial data: {e}")
            return pd.DataFrame()

    def analyze_parameter_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between parameters and performance."""
        if df.empty or 'objective_value' not in df.columns:
            return {}

        results = {
            "correlations": {},
            "top_performers": {},
            "parameter_sweet_spots": {},
            "interactions": [],
        }

        # Get numeric columns (parameters)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        param_cols = [c for c in numeric_cols if c not in ['trial_id', 'trial_number', 'objective_value']]

        # Calculate correlations with objective value
        for col in param_cols:
            if col in df.columns:
                try:
                    corr = df[col].astype(float).corr(df['objective_value'].astype(float))
                    if not np.isnan(corr):
                        results["correlations"][col] = round(corr, 4)
                except:
                    pass

        # Sort by absolute correlation
        results["correlations"] = dict(
            sorted(results["correlations"].items(),
                   key=lambda x: abs(x[1]),
                   reverse=True)
        )

        # Find sweet spots for top correlated parameters
        top_params = list(results["correlations"].keys())[:5]

        for param in top_params:
            if param in df.columns:
                try:
                    # Get top 10% performers
                    threshold = df['objective_value'].quantile(0.9)
                    top_df = df[df['objective_value'] >= threshold]

                    if len(top_df) > 0:
                        results["parameter_sweet_spots"][param] = {
                            "mean": round(float(top_df[param].mean()), 6),
                            "std": round(float(top_df[param].std()), 6),
                            "min": round(float(top_df[param].min()), 6),
                            "max": round(float(top_df[param].max()), 6),
                        }
                except:
                    pass

        # Analyze top performers
        top_n = min(20, len(df))
        top_trials = df.nlargest(top_n, 'objective_value')

        results["top_performers"] = {
            "count": top_n,
            "mean_value": round(float(top_trials['objective_value'].mean()), 6),
            "trial_numbers": top_trials['trial_number'].tolist(),
        }

        # Look for parameter interactions (simplified)
        if len(top_params) >= 2:
            for i, p1 in enumerate(top_params[:3]):
                for p2 in top_params[i+1:4]:
                    if p1 in df.columns and p2 in df.columns:
                        try:
                            # Check if high values of both correlate with performance
                            df['interaction'] = df[p1].astype(float) * df[p2].astype(float)
                            interaction_corr = df['interaction'].corr(df['objective_value'].astype(float))

                            if abs(interaction_corr) > 0.1:
                                results["interactions"].append({
                                    "params": [p1, p2],
                                    "correlation": round(interaction_corr, 4),
                                })
                        except:
                            pass

        return results

    def find_overlooked_trials(self, df: pd.DataFrame, current_ensemble: List[int]) -> List[Dict]:
        """Find high-performing trials not in current ensemble."""
        if df.empty:
            return []

        # Get trials not in ensemble
        non_ensemble = df[~df['trial_number'].isin(current_ensemble)]

        if non_ensemble.empty:
            return []

        # Find top performers not in ensemble
        threshold = df['objective_value'].quantile(0.95)
        overlooked = non_ensemble[non_ensemble['objective_value'] >= threshold]

        results = []
        for _, row in overlooked.nlargest(10, 'objective_value').iterrows():
            results.append({
                "trial_number": int(row['trial_number']),
                "value": round(float(row['objective_value']), 6),
                "rank_overall": int((df['objective_value'] >= row['objective_value']).sum()),
            })

        return results

    def generate_ai_insights(self, analysis_data: Dict) -> str:
        """Use Ollama to generate insights from the analysis."""
        prompt = f"""You are an AI assistant analyzing historical deep reinforcement learning training trials for cryptocurrency trading.

ANALYSIS DATA:
{json.dumps(analysis_data, indent=2)}

Based on this analysis of archived training trials, provide:

1. KEY FINDINGS: What are the most important patterns you see in the data?

2. OVERLOOKED OPPORTUNITIES: Based on the "overlooked_trials" data, are there high-performing configurations that should be reconsidered?

3. PARAMETER RECOMMENDATIONS: Based on the correlations and sweet spots, what specific parameter ranges should future training focus on?

4. INTERACTIONS TO EXPLORE: What parameter combinations seem promising based on the interaction data?

5. ACTIONABLE SUGGESTIONS: Provide 3 specific, actionable recommendations for improving future training runs.

Be specific and data-driven in your analysis. Reference actual values from the data."""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 2000,
                    }
                },
                timeout=120,
            )

            if response.status_code == 200:
                return response.json().get("response", "No response generated")
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return f"Error: {response.status_code}"

        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return f"Error: {e}"

    def run_full_analysis(self, use_ai: bool = True, study_name: str = None) -> Dict[str, Any]:
        """Run full analysis on archived trials.

        Args:
            use_ai: Whether to generate AI insights
            study_name: Optional study name to filter (use % as wildcard)
        """
        logger.info("="*60)
        logger.info("ARCHIVE ANALYSIS")
        logger.info("="*60)

        # Load archive manifest
        manifest = self.load_manifest()
        archived_trials = self.get_archived_trial_numbers()

        logger.info(f"Found {len(archived_trials)} archived trials in {len(manifest.get('batches', {}))} batches")

        # Get current ensemble and its study
        ensemble_path = Path("train_results/ensemble/ensemble_manifest.json")
        current_ensemble = []
        ensemble_study = None
        if ensemble_path.exists():
            with open(ensemble_path) as f:
                manifest_data = json.load(f)
                current_ensemble = manifest_data.get("trial_numbers", [])
                ensemble_study = manifest_data.get("study_name")

        logger.info(f"Current ensemble has {len(current_ensemble)} trials")
        if ensemble_study:
            logger.info(f"Ensemble study: {ensemble_study}")

        # Use ensemble study if no study specified
        if not study_name and ensemble_study:
            # Use pattern to match related studies
            base_study = ensemble_study.rsplit('_', 2)[0] if ensemble_study else None
            if base_study:
                study_name = f"{base_study}%"
                logger.info(f"Auto-filtering to studies matching: {study_name}")

        # Load all trial data from database
        logger.info("Loading trial data from database...")
        df = self.get_trial_data_from_db(study_name=study_name)

        if df.empty:
            logger.error("No trial data found in database")
            return {"error": "No trial data"}

        logger.info(f"Loaded {len(df)} trials from database")

        # Filter to archived trials for focused analysis
        df_archived = df[df['trial_number'].isin(archived_trials)]
        logger.info(f"Analyzing {len(df_archived)} archived trials")

        # Run analysis
        logger.info("Analyzing parameter correlations...")
        correlations = self.analyze_parameter_correlations(df)  # Use all data

        logger.info("Finding overlooked high-performers...")
        overlooked = self.find_overlooked_trials(df, current_ensemble)

        # Compile results
        analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "total_trials_analyzed": len(df),
            "archived_trials": len(archived_trials),
            "current_ensemble_size": len(current_ensemble),
            "correlations": correlations.get("correlations", {}),
            "parameter_sweet_spots": correlations.get("parameter_sweet_spots", {}),
            "interactions": correlations.get("interactions", []),
            "top_performers": correlations.get("top_performers", {}),
            "overlooked_trials": overlooked,
            "statistics": {
                "mean_value": round(float(df['objective_value'].mean()), 6) if 'objective_value' in df else None,
                "std_value": round(float(df['objective_value'].std()), 6) if 'objective_value' in df else None,
                "best_value": round(float(df['objective_value'].max()), 6) if 'objective_value' in df else None,
                "worst_value": round(float(df['objective_value'].min()), 6) if 'objective_value' in df else None,
            }
        }

        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS RESULTS")
        print("="*60)

        print(f"\nTrials analyzed: {len(df)}")
        print(f"Archived trials: {len(archived_trials)}")

        print("\nTOP PARAMETER CORRELATIONS (with performance):")
        for param, corr in list(analysis_results["correlations"].items())[:10]:
            direction = "+" if corr > 0 else ""
            print(f"  {param}: {direction}{corr:.4f}")

        print("\nPARAMETER SWEET SPOTS (top 10% performers):")
        for param, stats in analysis_results["parameter_sweet_spots"].items():
            print(f"  {param}: {stats['mean']:.6f} (range: {stats['min']:.6f} - {stats['max']:.6f})")

        if overlooked:
            print(f"\nOVERLOOKED HIGH-PERFORMERS (not in ensemble):")
            for trial in overlooked[:5]:
                print(f"  Trial {trial['trial_number']}: value={trial['value']:.6f} (rank #{trial['rank_overall']})")

        if analysis_results["interactions"]:
            print("\nPROMISING PARAMETER INTERACTIONS:")
            for interaction in analysis_results["interactions"][:5]:
                print(f"  {interaction['params'][0]} x {interaction['params'][1]}: {interaction['correlation']:.4f}")

        # Generate AI insights
        if use_ai:
            print("\n" + "="*60)
            print("AI INSIGHTS")
            print("="*60)
            logger.info("Generating AI insights...")

            ai_insights = self.generate_ai_insights(analysis_results)
            analysis_results["ai_insights"] = ai_insights
            print(ai_insights)

        # Save results
        output_path = Path(f"analysis_reports/archive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)

        logger.info(f"Analysis saved to: {output_path}")

        return analysis_results

    def export_to_csv(self) -> Path:
        """Export all trial data to CSV for manual analysis."""
        df = self.get_trial_data_from_db()

        if df.empty:
            logger.error("No data to export")
            return None

        output_path = Path(f"analysis_reports/all_trials_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} trials to: {output_path}")

        return output_path


def main():
    parser = argparse.ArgumentParser(description="Analyze archived training trials")
    parser.add_argument("--analyze", action="store_true", help="Run full analysis")
    parser.add_argument("--no-ai", action="store_true", help="Skip AI insights generation")
    parser.add_argument("--export-csv", action="store_true", help="Export trial data to CSV")
    parser.add_argument("--archive-dir", type=str, default="/var/tmp/cappuccino_archive",
                        help="Archive directory")
    parser.add_argument("--model", type=str, default="qwen2.5-coder:7b",
                        help="Ollama model for AI insights")
    parser.add_argument("--study", type=str, default=None,
                        help="Study name to filter (use %% as wildcard, e.g. 'cappuccino_fresh%%')")
    parser.add_argument("--all-studies", action="store_true",
                        help="Analyze all studies (don't auto-filter to ensemble study)")

    args = parser.parse_args()

    analyzer = ArchiveAnalyzer(
        archive_dir=args.archive_dir,
        ollama_model=args.model,
    )

    if args.export_csv:
        analyzer.export_to_csv()
    elif args.analyze:
        study = args.study if args.study else (None if args.all_studies else None)
        analyzer.run_full_analysis(use_ai=not args.no_ai, study_name=study)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
