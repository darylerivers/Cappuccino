"""
Cappuccino Performance Report Generator
Collects live trading data and gets an analysis from Claude Opus.

Usage:
    python ~/cappuccino/report.py
    python ~/cappuccino/report.py --since 2026-03-01
    python ~/cappuccino/report.py --save report.md
"""

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import anthropic
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE         = Path.home() / "cappuccino"
MOMENTUM_DIR = BASE / "momentum"
XGB_DIR      = BASE / "xgboost"

STATE_FILE   = MOMENTUM_DIR / "state.json"
CFG_FILE     = MOMENTUM_DIR / "config.yaml"
AUDIT_DB     = MOMENTUM_DIR / "audit.db"
VERSION_FILE = MOMENTUM_DIR / "version.json"
METRICS_FILE = XGB_DIR / "results" / "metrics.json"


# ── Data collectors ────────────────────────────────────────────────────────────

def load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def load_yaml(p: Path) -> dict:
    try:
        import yaml
        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


def collect_portfolio(since_dt: datetime | None) -> dict:
    state = load_json(STATE_FILE)
    cfg   = load_yaml(CFG_FILE)
    ver   = load_json(VERSION_FILE)

    pnl_hist = state.get("pnl_history", [])
    if since_dt:
        pnl_hist = [r for r in pnl_hist
                    if datetime.fromisoformat(r["ts"]) >= since_dt]

    inception = float(state.get("inception_value") or cfg.get("capital_usdt", 500))
    peak      = float(state.get("peak_value") or inception)

    latest_val = float(pnl_hist[-1]["value"]) if pnl_hist else inception
    pnl_pct    = (latest_val / inception - 1) * 100 if inception else 0
    dd_pct     = (latest_val / peak - 1) * 100 if peak else 0

    return {
        "strategy_version":  ver.get("version", "unknown"),
        "deployed":          ver.get("deployed", "unknown"),
        "notes":             ver.get("notes", ""),
        "exchange":          cfg.get("exchange", "coinbase"),
        "paper_mode":        cfg.get("paper_mode", True),
        "capital_inception": inception,
        "capital_current":   latest_val,
        "peak_value":        peak,
        "pnl_pct":           round(pnl_pct, 4),
        "drawdown_pct":      round(dd_pct, 4),
        "rebal_count":       state.get("rebal_count", 0),
        "circuit_breaker":   state.get("circuit_breaker"),
        "last_regime":       state.get("last_regime"),
        "positions":         state.get("positions", {}),
        "regime_ma_bars":    cfg.get("regime_ma_bars", 168),
        "lookback_bars":     cfg.get("lookback_bars", 24),
        "top_n":             cfg.get("top_n", 2),
        "max_drawdown_limit": cfg.get("max_drawdown_pct", 15),
        "assets":            cfg.get("assets", []),
        "pnl_history":       pnl_hist,
    }


def collect_audit(since_dt: datetime | None) -> dict:
    if not AUDIT_DB.exists():
        return {"trades": [], "snapshots": [], "events": []}

    conn = sqlite3.connect(f"file:{AUDIT_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    def _since_filter(col: str) -> tuple[str, list]:
        if since_dt:
            return f"WHERE {col} >= ?", [since_dt.isoformat()]
        return "", []

    def _q(table: str, ts_col: str, limit: int) -> list[dict]:
        where, params = _since_filter(ts_col)
        rows = conn.execute(
            f"SELECT * FROM {table} {where} ORDER BY id DESC LIMIT {limit}",
            params
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    try:
        trades    = _q("trades",    "ts", 50)
        snapshots = _q("snapshots", "ts", 24)
        events    = _q("events",    "ts", 30)
    except Exception:
        trades = snapshots = events = []

    conn.close()
    return {"trades": trades, "snapshots": snapshots, "events": events}


def collect_xgb() -> dict:
    m = load_json(METRICS_FILE)
    if not m:
        return {"available": False}

    summary = {
        "available":         True,
        "timestamp":         m.get("timestamp"),
        "verdict":           m.get("verdict"),
        "combined_oos_auc":  round(m.get("combined_oos_auc", 0), 4),
        "combined_accuracy": round(m.get("combined_oos_accuracy", 0), 4),
        "per_asset":         {},
    }

    for asset, folds in m.get("per_asset", {}).items():
        avg_auc = round(sum(f["auc"] for f in folds) / len(folds), 4) if folds else 0
        summary["per_asset"][asset] = {
            "avg_auc":  avg_auc,
            "n_folds":  len(folds),
            "folds":    [{"fold": f["fold"], "test_period": f["test_period"],
                          "auc": f["auc"], "accuracy": f["accuracy"]} for f in folds],
        }

    return summary


# ── Prompt builder ─────────────────────────────────────────────────────────────

SYSTEM = """\
You are a quantitative trading analyst reviewing a live algorithmic trading system.
The system is called Cappuccino — a momentum + regime-filter strategy running on Coinbase.

Be direct, precise, and actionable. Use markdown. Structure your report with:
1. Executive Summary (3-5 bullets)
2. Performance Analysis
3. Risk Assessment
4. XGBoost Signal Quality
5. Operational Health
6. Recommendations

Flag anything that needs immediate attention with ⚠️.
Flag positives with ✅.
"""


def build_prompt(portfolio: dict, audit: dict, xgb: dict, since_label: str) -> str:
    return f"""\
Please generate a comprehensive performance report for the Cappuccino trading system.
Period covered: {since_label}

## Strategy Configuration
```json
{json.dumps({k: v for k, v in portfolio.items()
             if k not in ("pnl_history", "positions")}, indent=2)}
```

## Current Positions
```json
{json.dumps(portfolio["positions"], indent=2)}
```

## P&L History ({len(portfolio["pnl_history"])} data points)
```json
{json.dumps(portfolio["pnl_history"], indent=2)}
```

## Recent Trades ({len(audit["trades"])} shown)
```json
{json.dumps(audit["trades"], indent=2)}
```

## Recent Snapshots ({len(audit["snapshots"])} shown)
```json
{json.dumps(audit["snapshots"], indent=2)}
```

## Audit Events ({len(audit["events"])} shown)
```json
{json.dumps(audit["events"], indent=2)}
```

## XGBoost Signal Quality
```json
{json.dumps(xgb, indent=2)}
```

Please analyze all of the above and produce the structured report.
"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cappuccino performance report via Claude Opus")
    parser.add_argument("--since",  default=None,
                        help="Start date YYYY-MM-DD (default: last 7 days)")
    parser.add_argument("--save",   default=None,
                        help="Save report to file (e.g. report.md)")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable extended thinking (faster, cheaper)")
    args = parser.parse_args()

    if args.since:
        since_dt    = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
        since_label = f"since {args.since}"
    else:
        since_dt    = datetime.now(timezone.utc) - timedelta(days=7)
        since_label = "last 7 days"

    print(f"Collecting data ({since_label})...")
    portfolio = collect_portfolio(since_dt)
    audit     = collect_audit(since_dt)
    xgb       = collect_xgb()

    print(f"  Portfolio: ${portfolio['capital_current']:,.2f} "
          f"({portfolio['pnl_pct']:+.2f}%  dd {portfolio['drawdown_pct']:+.2f}%)")
    print(f"  Trades: {len(audit['trades'])}  |  Snapshots: {len(audit['snapshots'])}  "
          f"|  Events: {len(audit['events'])}")
    print(f"  XGBoost AUC: {xgb.get('combined_oos_auc', 'N/A')}  "
          f"verdict: {xgb.get('verdict', 'N/A')}")
    print()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        print("  Export it:  export ANTHROPIC_API_KEY=sk-ant-...")
        return

    client = anthropic.Anthropic(api_key=api_key)
    prompt = build_prompt(portfolio, audit, xgb, since_label)

    create_kwargs = dict(
        model="claude-opus-4-6",
        max_tokens=8192,
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    if not args.no_think:
        create_kwargs["thinking"] = {"type": "adaptive"}

    print(f"Generating report with Claude Opus 4.6"
          f"{' + adaptive thinking' if not args.no_think else ''}...\n")
    print("─" * 72)

    report_chunks = []

    with client.messages.stream(**create_kwargs) as stream:
        for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "thinking":
                    print("\n[thinking...]\n", flush=True)
                elif event.content_block.type == "text":
                    pass  # start of report text
            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    chunk = event.delta.text
                    print(chunk, end="", flush=True)
                    report_chunks.append(chunk)

        final = stream.get_final_message()

    report_text = "".join(report_chunks)

    print("\n" + "─" * 72)
    print(f"\nTokens used: {final.usage.input_tokens:,} in / "
          f"{final.usage.output_tokens:,} out")

    if args.save:
        save_path = Path(args.save)
        header = (f"# Cappuccino Performance Report\n"
                  f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
                  f"Period: {since_label}  |  Model: claude-opus-4-6\n\n---\n\n")
        save_path.write_text(header + report_text)
        print(f"\nReport saved to {save_path}")


if __name__ == "__main__":
    main()
