#!/usr/bin/env python3
"""
Ollama-powered training data analyzer and hyperparameter advisor.

This script analyzes Optuna training results and uses a locally-run Ollama model
to provide insights and suggest parameter improvements.

Usage:
    python ollama_training_advisor.py --study cappuccino_3workers_20251102_2325
    python ollama_training_advisor.py --study cappuccino_3workers_20251102_2325 --model qwen2.5-coder:7b
    python ollama_training_advisor.py --study cappuccino_3workers_20251102_2325 --web-search
"""

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import requests


class OllamaTrainingAdvisor:
    """Analyzes training data and provides AI-powered recommendations."""

    def __init__(
        self,
        db_path: str,
        study_name: str,
        ollama_model: str = "mistral:latest",
        enable_web_search: bool = False,
        ollama_host: str = "http://localhost:11434",
    ):
        self.db_path = db_path
        self.study_name = study_name
        self.ollama_model = ollama_model
        self.enable_web_search = enable_web_search
        self.ollama_host = ollama_host

        self.study_data = {}
        self.insights = {}

    def check_ollama(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available = [m["name"] for m in models]
                if self.ollama_model in available:
                    print(f"✓ Ollama is running with model '{self.ollama_model}'")
                    return True
                else:
                    print(f"✗ Model '{self.ollama_model}' not found. Available: {available}")
                    print(f"  Pull it with: ollama pull {self.ollama_model}")
                    return False
        except requests.exceptions.RequestException as e:
            print(f"✗ Ollama is not running or not accessible: {e}")
            print("  Start it with: ollama serve")
            return False

    def load_training_data(self) -> Dict[str, Any]:
        """Load training data from Optuna database."""
        print(f"\nLoading data from study '{self.study_name}'...")

        conn = sqlite3.connect(self.db_path)

        # Get study ID
        study_df = pd.read_sql_query(
            "SELECT study_id FROM studies WHERE study_name = ?",
            conn,
            params=(self.study_name,)
        )

        if study_df.empty:
            print(f"Study '{self.study_name}' not found")
            conn.close()
            return {}

        study_id = int(study_df['study_id'].iloc[0])

        # Get trials with values
        trials_query = """
        SELECT
            t.trial_id,
            t.number,
            t.state,
            t.datetime_start,
            t.datetime_complete,
            tv.value
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = ?
        ORDER BY t.number
        """
        trials_df = pd.read_sql_query(trials_query, conn, params=(study_id,))

        print(f"DEBUG: Found {len(trials_df)} total trials for study_id {study_id}")

        # Get parameters
        params_query = """
        SELECT
            t.number,
            tp.param_name,
            tp.param_value
        FROM trial_params tp
        JOIN trials t ON tp.trial_id = t.trial_id
        WHERE t.study_id = ?
        """
        params_df = pd.read_sql_query(params_query, conn, params=(study_id,))

        # Get user attributes (model names, etc.)
        attrs_query = """
        SELECT
            t.number,
            tua.key,
            tua.value_json
        FROM trial_user_attributes tua
        JOIN trials t ON tua.trial_id = t.trial_id
        WHERE t.study_id = ?
        """
        attrs_df = pd.read_sql_query(attrs_query, conn, params=(study_id,))

        conn.close()

        # Filter completed trials with values
        completed = trials_df[(trials_df['state'] == 'COMPLETE') & (trials_df['value'].notna())].copy()

        if completed.empty:
            print("No completed trials with values found")
            return {}

        # Calculate durations
        completed['datetime_start'] = pd.to_datetime(completed['datetime_start'])
        completed['datetime_complete'] = pd.to_datetime(completed['datetime_complete'])
        completed['duration_minutes'] = (
            completed['datetime_complete'] - completed['datetime_start']
        ).dt.total_seconds() / 60

        # Extract model names
        model_map = {}
        for _, row in attrs_df.iterrows():
            if row['key'] == 'model_name':
                model_map[row['number']] = row['value_json'].strip('"')

        completed['model'] = completed['number'].map(model_map)

        # Organize parameters by trial
        params_by_trial = {}
        for _, row in params_df.iterrows():
            trial_num = row['number']
            if trial_num not in params_by_trial:
                params_by_trial[trial_num] = {}
            params_by_trial[trial_num][row['param_name']] = row['param_value']

        # Calculate statistics
        values = completed['value'].dropna()

        self.study_data = {
            'trials_df': completed,
            'params_by_trial': params_by_trial,
            'total_trials': len(trials_df),
            'completed_trials': len(completed),
            'failed_trials': len(trials_df[trials_df['state'] == 'FAIL']),
            'stats': {
                'mean': values.mean(),
                'median': values.median(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'q1': values.quantile(0.25),
                'q3': values.quantile(0.75),
            }
        }

        print(f"✓ Loaded {len(completed)} completed trials")
        print(f"  Performance range: {values.min():.6f} to {values.max():.6f}")

        return self.study_data

    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in hyperparameter performance."""
        print("\nAnalyzing hyperparameter patterns...")

        completed = self.study_data['trials_df']
        params_by_trial = self.study_data['params_by_trial']

        # Get top 10 and bottom 10 trials
        top_10 = completed.nlargest(10, 'value')
        bottom_10 = completed.nsmallest(10, 'value')

        # Collect parameters for top and bottom trials
        top_params = []
        for trial_num in top_10['number']:
            if trial_num in params_by_trial:
                top_params.append(params_by_trial[trial_num])

        bottom_params = []
        for trial_num in bottom_10['number']:
            if trial_num in params_by_trial:
                bottom_params.append(params_by_trial[trial_num])

        # Calculate parameter statistics
        param_stats = {}
        if top_params:
            all_param_names = set(top_params[0].keys())
            for param_name in all_param_names:
                top_values = [p[param_name] for p in top_params if param_name in p]
                bottom_values = [p[param_name] for p in bottom_params if param_name in p]

                if top_values and bottom_values:
                    param_stats[param_name] = {
                        'top_mean': np.mean(top_values),
                        'top_std': np.std(top_values),
                        'bottom_mean': np.mean(bottom_values),
                        'bottom_std': np.std(bottom_values),
                        'difference': np.mean(top_values) - np.mean(bottom_values),
                    }

        # Model performance
        model_perf = {}
        if 'model' in completed.columns:
            model_stats = completed.groupby('model')['value'].agg(['count', 'mean', 'std', 'min', 'max'])
            model_perf = model_stats.to_dict('index')

        self.insights = {
            'top_trials': top_10[['number', 'value', 'model', 'duration_minutes']].to_dict('records'),
            'bottom_trials': bottom_10[['number', 'value', 'model', 'duration_minutes']].to_dict('records'),
            'top_params': top_params,
            'bottom_params': bottom_params,
            'param_stats': param_stats,
            'model_performance': model_perf,
        }

        print(f"✓ Analyzed {len(param_stats)} hyperparameters")
        return self.insights

    def search_web(self, query: str) -> str:
        """Search the web for information (basic implementation)."""
        # Simple DuckDuckGo search using requests
        # Note: This is a basic implementation. For production, consider using proper APIs
        try:
            search_url = f"https://lite.duckduckgo.com/lite/?q={query.replace(' ', '+')}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code == 200:
                return f"Web search completed for: {query}\n(Results would be parsed from DuckDuckGo)"
            return "Web search unavailable"
        except Exception as e:
            return f"Web search error: {e}"

    def query_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Query Ollama model with a prompt."""
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
                timeout=120
            )

            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error: {response.status_code}"

        except Exception as e:
            return f"Error querying Ollama: {e}"

    def generate_analysis_report(self) -> str:
        """Generate detailed analysis using Ollama."""
        print("\n" + "="*80)
        print("GENERATING AI ANALYSIS (this may take 30-60 seconds)...")
        print("="*80)

        # Prepare data summary for the AI
        data_summary = f"""
TRAINING STUDY: {self.study_name}

OVERALL PERFORMANCE:
- Total trials: {self.study_data['total_trials']}
- Completed: {self.study_data['completed_trials']}
- Failed: {self.study_data['failed_trials']}
- Best value: {self.study_data['stats']['max']:.6f}
- Worst value: {self.study_data['stats']['min']:.6f}
- Mean: {self.study_data['stats']['mean']:.6f}
- Median: {self.study_data['stats']['median']:.6f}
- Std Dev: {self.study_data['stats']['std']:.6f}

TOP 5 PERFORMING TRIALS:
"""
        for i, trial in enumerate(self.insights['top_trials'][:5], 1):
            data_summary += f"{i}. Trial #{trial['number']}: {trial['value']:.6f} ({trial.get('model', 'unknown')})\n"

        data_summary += "\nHYPERPARAMETER ANALYSIS:\n"
        for param_name, stats in sorted(
            self.insights['param_stats'].items(),
            key=lambda x: abs(x[1]['difference']),
            reverse=True
        )[:10]:
            data_summary += f"- {param_name}:\n"
            data_summary += f"  Top performers avg: {stats['top_mean']:.6f} (±{stats['top_std']:.4f})\n"
            data_summary += f"  Bottom performers avg: {stats['bottom_mean']:.6f} (±{stats['bottom_std']:.4f})\n"
            data_summary += f"  Difference: {stats['difference']:.6f}\n"

        if self.insights['model_performance']:
            data_summary += "\nMODEL PERFORMANCE:\n"
            for model, stats in self.insights['model_performance'].items():
                data_summary += f"- {model}: avg={stats['mean']:.6f}, trials={stats['count']}\n"

        # Create the prompt for Ollama
        system_prompt = """You are an expert in deep reinforcement learning and hyperparameter optimization.
You specialize in analyzing training results and suggesting improvements for DRL agents in financial trading environments.
Your expertise includes PPO, SAC, DDPG, TD3, and A2C algorithms."""

        prompt = f"""{data_summary}

Based on this training data for a cryptocurrency trading DRL agent, please provide:

1. KEY INSIGHTS: What patterns do you notice in the hyperparameters? Which parameters seem most impactful?

2. POTENTIAL ISSUES: Are there any red flags or concerns in the training results?

3. RECOMMENDATIONS: Suggest 3-5 specific hyperparameter changes or ranges to explore next.

4. SEARCH SPACE REFINEMENT: Should we narrow or expand any parameter ranges?

5. ALGORITHMIC SUGGESTIONS: Are there alternative approaches or techniques worth trying?

Please be specific and actionable. Focus on concrete suggestions that could improve performance.
"""

        # Query Ollama
        response = self.query_ollama(prompt, system_prompt)

        return response

    def generate_web_enhanced_analysis(self) -> str:
        """Generate analysis enhanced with web search results."""
        print("\nSearching web for latest DRL best practices...")

        # Search for recent techniques
        search_queries = [
            "PPO hyperparameter tuning deep reinforcement learning 2024",
            "cryptocurrency trading DRL best practices",
            "financial time series reinforcement learning optimization",
        ]

        web_context = "RECENT WEB SEARCH INSIGHTS:\n\n"
        for query in search_queries:
            result = self.search_web(query)
            web_context += f"Query: {query}\n{result}\n\n"

        # Add web context to the analysis
        system_prompt = """You are an expert in deep reinforcement learning with access to recent research.
Analyze the training data and incorporate latest best practices from the web search results."""

        data_summary = self.generate_analysis_report()

        prompt = f"""{web_context}

TRAINING DATA:
{data_summary}

Given the training results AND the latest research insights from web searches, provide:
1. How do our results compare to state-of-the-art?
2. Any recent techniques we should incorporate?
3. Specific, actionable recommendations based on latest research.
"""

        return self.query_ollama(prompt, system_prompt)

    def save_recommendations(self, analysis: str, output_file: str):
        """Save analysis to file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open('w') as f:
            f.write("="*80 + "\n")
            f.write(f"OLLAMA TRAINING ANALYSIS REPORT\n")
            f.write(f"Study: {self.study_name}\n")
            f.write(f"Model: {self.ollama_model}\n")
            f.write(f"Date: {pd.Timestamp.now()}\n")
            f.write("="*80 + "\n\n")
            f.write(analysis)

        print(f"\n✓ Report saved to: {output_file}")

    def run(self):
        """Run the full analysis pipeline."""
        # Check Ollama
        if not self.check_ollama():
            return

        # Load data
        if not self.load_training_data():
            return

        # Analyze patterns
        self.analyze_patterns()

        # Generate analysis
        if self.enable_web_search:
            analysis = self.generate_web_enhanced_analysis()
        else:
            analysis = self.generate_analysis_report()

        # Print analysis
        print("\n" + "="*80)
        print("OLLAMA ANALYSIS & RECOMMENDATIONS")
        print("="*80)
        print(analysis)
        print("="*80 + "\n")

        # Save to file
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"analysis_reports/ollama_analysis_{self.study_name}_{timestamp}.txt"
        self.save_recommendations(analysis, output_file)

        return analysis


def main():
    parser = argparse.ArgumentParser(
        description="AI-powered training analysis using Ollama"
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
        "--model",
        type=str,
        default="mistral:latest",
        help="Ollama model to use (default: mistral:latest)"
    )
    parser.add_argument(
        "--web-search",
        action="store_true",
        help="Enable web search for latest research"
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama API host (default: http://localhost:11434)"
    )

    args = parser.parse_args()

    advisor = OllamaTrainingAdvisor(
        db_path=args.db,
        study_name=args.study,
        ollama_model=args.model,
        enable_web_search=args.web_search,
        ollama_host=args.ollama_host,
    )

    advisor.run()


if __name__ == "__main__":
    main()
