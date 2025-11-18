#!/usr/bin/env python3
"""
Advanced Ollama-powered hyperparameter suggester.

This script not only analyzes training results but also generates new hyperparameter
configurations to try, based on AI analysis and optionally web research.

Usage:
    python ollama_param_suggester.py --study cappuccino_3workers_20251102_2325 --generate 5
    python ollama_param_suggester.py --study cappuccino_3workers_20251102_2325 --generate 10 --web-search
"""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import requests


class OllamaParamSuggester:
    """Generates new hyperparameter configurations using AI analysis."""

    def __init__(
        self,
        db_path: str,
        study_name: str,
        ollama_model: str = "qwen2.5-coder:7b",
        ollama_host: str = "http://localhost:11434",
    ):
        self.db_path = db_path
        self.study_name = study_name
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host
        self.study_data = {}

    def load_study_data(self):
        """Load and analyze study data."""
        print(f"Loading study '{self.study_name}'...")

        conn = sqlite3.connect(self.db_path)

        # Get study ID
        study_df = pd.read_sql_query(
            "SELECT study_id FROM studies WHERE study_name = ?",
            conn,
            params=(self.study_name,)
        )

        if study_df.empty:
            print(f"Study not found")
            return False

        study_id = int(study_df['study_id'].iloc[0])

        # Get completed trials with values
        trials_query = """
        SELECT t.trial_id, t.number, tv.value
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = ? AND t.state = 'COMPLETE' AND tv.value IS NOT NULL
        ORDER BY t.number
        """
        trials_df = pd.read_sql_query(trials_query, conn, params=(study_id,))

        # Get all parameters
        params_query = """
        SELECT t.number, tp.param_name, tp.param_value
        FROM trial_params tp
        JOIN trials t ON tp.trial_id = t.trial_id
        WHERE t.study_id = ?
        """
        params_df = pd.read_sql_query(params_query, conn, params=(study_id,))
        conn.close()

        # Organize by trial
        params_by_trial = {}
        for _, row in params_df.iterrows():
            trial_num = row['number']
            if trial_num not in params_by_trial:
                params_by_trial[trial_num] = {}
            params_by_trial[trial_num][row['param_name']] = row['param_value']

        # Merge with trial values
        trial_data = []
        for _, trial in trials_df.iterrows():
            if trial['number'] in params_by_trial:
                trial_info = params_by_trial[trial['number']].copy()
                trial_info['_value'] = trial['value']
                trial_info['_trial'] = trial['number']
                trial_data.append(trial_info)

        self.study_data = {
            'trials': pd.DataFrame(trial_data),
            'param_names': list(params_by_trial[list(params_by_trial.keys())[0]].keys()) if params_by_trial else [],
        }

        print(f"✓ Loaded {len(trial_data)} trials with {len(self.study_data['param_names'])} parameters")
        return True

    def analyze_param_distributions(self) -> Dict[str, Dict]:
        """Analyze parameter distributions and correlations."""
        df = self.study_data['trials']

        # Get top 20% performers
        threshold = df['_value'].quantile(0.8)
        top_performers = df[df['_value'] >= threshold]

        param_analysis = {}
        for param in self.study_data['param_names']:
            all_values = df[param].dropna()
            top_values = top_performers[param].dropna()

            param_analysis[param] = {
                'all_mean': float(all_values.mean()),
                'all_std': float(all_values.std()),
                'all_min': float(all_values.min()),
                'all_max': float(all_values.max()),
                'top_mean': float(top_values.mean()),
                'top_std': float(top_values.std()),
                'top_min': float(top_values.min()),
                'top_max': float(top_values.max()),
                'correlation': float(df[[param, '_value']].corr().iloc[0, 1]),
            }

        return param_analysis

    def query_ollama(self, prompt: str, system_prompt: str = None) -> str:
        """Query Ollama model."""
        try:
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
            }
            if system_prompt:
                payload["system"] = system_prompt

            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=180
            )

            if response.status_code == 200:
                return response.json()['response']
            return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {e}"

    def generate_suggestions(self, num_configs: int = 5) -> List[Dict]:
        """Generate new parameter configurations using AI."""
        print(f"\nGenerating {num_configs} new configurations using {self.ollama_model}...")

        # Analyze current distributions
        param_analysis = self.analyze_param_distributions()

        # Get top 5 trials
        df = self.study_data['trials']
        top_5 = df.nlargest(5, '_value')

        # Prepare context for AI
        context = f"""TRAINING STUDY ANALYSIS:

Best performance: {df['_value'].max():.6f}
Top 20% threshold: {df['_value'].quantile(0.8):.6f}
Mean performance: {df['_value'].mean():.6f}

TOP 5 CONFIGURATIONS:
"""
        for idx, row in top_5.iterrows():
            context += f"\nTrial {row['_trial']}: {row['_value']:.6f}\n"
            for param in self.study_data['param_names']:
                context += f"  {param}: {row[param]}\n"

        context += "\n\nPARAMETER STATISTICS:\n"
        for param, stats in param_analysis.items():
            context += f"\n{param}:\n"
            context += f"  Overall: mean={stats['all_mean']:.4f}, std={stats['all_std']:.4f}, range=[{stats['all_min']:.4f}, {stats['all_max']:.4f}]\n"
            context += f"  Top 20%: mean={stats['top_mean']:.4f}, std={stats['top_std']:.4f}, range=[{stats['top_min']:.4f}, {stats['top_max']:.4f}]\n"
            context += f"  Correlation with performance: {stats['correlation']:.4f}\n"

        # Create AI prompt
        system_prompt = """You are an expert in hyperparameter optimization for deep reinforcement learning.
You understand PPO, SAC, DDPG, TD3, and A2C algorithms in financial trading contexts.
You can identify promising parameter combinations and suggest intelligent search directions."""

        prompt = f"""{context}

Based on this data, generate {num_configs} NEW hyperparameter configurations that are likely to perform well.

REQUIREMENTS:
1. Focus on parameters with high correlation to performance
2. Explore around the ranges where top performers cluster
3. Try some variations that might discover better regions
4. Balance exploitation (near best) and exploration (new regions)
5. Output ONLY valid JSON array format

Output format (MUST be valid JSON):
[
  {{
    "rationale": "Why this config might work",
    "learning_rate": 0.0001,
    "batch_size": 512,
    "gamma": 0.99,
    ...
  }},
  ...
]

Generate {num_configs} configurations now:"""

        # Query AI
        response = self.query_ollama(prompt, system_prompt)

        # Parse response
        try:
            # Try to extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                suggestions = json.loads(json_str)
                print(f"✓ Generated {len(suggestions)} configurations")
                return suggestions
            else:
                print("Could not find JSON in response")
                return []
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Response:\n{response[:500]}")
            return []

    def save_suggestions(self, suggestions: List[Dict], output_file: str):
        """Save suggestions to file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open('w') as f:
            json.dump(suggestions, f, indent=2)

        print(f"\n✓ Saved suggestions to: {output_file}")

    def print_suggestions(self, suggestions: List[Dict]):
        """Print suggestions in readable format."""
        print("\n" + "="*80)
        print("AI-GENERATED HYPERPARAMETER SUGGESTIONS")
        print("="*80)

        for i, config in enumerate(suggestions, 1):
            print(f"\n--- Configuration {i} ---")
            if 'rationale' in config:
                print(f"Rationale: {config['rationale']}")
                print()

            for key, value in config.items():
                if key != 'rationale':
                    print(f"  {key}: {value}")

        print("\n" + "="*80)

    def run(self, num_configs: int = 5):
        """Run the suggestion generation."""
        # Load data
        if not self.load_study_data():
            return

        # Generate suggestions
        suggestions = self.generate_suggestions(num_configs)

        if not suggestions:
            print("Failed to generate suggestions")
            return

        # Print
        self.print_suggestions(suggestions)

        # Save
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"analysis_reports/ollama_suggestions_{self.study_name}_{timestamp}.json"
        self.save_suggestions(suggestions, output_file)

        return suggestions


def main():
    parser = argparse.ArgumentParser(
        description="Generate new hyperparameter configurations using Ollama AI"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="databases/optuna_cappuccino.db",
        help="Path to Optuna database"
    )
    parser.add_argument(
        "--study",
        type=str,
        required=True,
        help="Study name to analyze"
    )
    parser.add_argument(
        "--generate",
        type=int,
        default=5,
        help="Number of configurations to generate (default: 5)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5-coder:7b",
        help="Ollama model to use (default: qwen2.5-coder:7b)"
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama API host"
    )

    args = parser.parse_args()

    suggester = OllamaParamSuggester(
        db_path=args.db,
        study_name=args.study,
        ollama_model=args.model,
        ollama_host=args.ollama_host,
    )

    suggester.run(num_configs=args.generate)


if __name__ == "__main__":
    main()
