"""
Momentum Strategy CLI Dashboard
Run: python ~/cappuccino/momentum/dashboard.py
Refreshes every 30 seconds.
"""

import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

BASE_DIR   = Path(__file__).parent
STATE_FILE = BASE_DIR / "state.json"
LOG_DIR    = BASE_DIR / "logs"
CFG_FILE   = BASE_DIR / "config.yaml"

REFRESH_SECS = 30


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def load_config() -> dict:
    if CFG_FILE.exists():
        import yaml
        with open(CFG_FILE) as f:
            return yaml.safe_load(f)
    return {}


def load_recent_logs(n: int = 50) -> list[dict]:
    """Load last n JSON log events from today's (and yesterday's) log file."""
    events = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    for date_str in [today, yesterday]:
        lf = LOG_DIR / f"{date_str}.jsonl"
        if lf.exists():
            with open(lf) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
    return events[-n:]


def fmt_pct(v: float, color_threshold: float = 0.0) -> Text:
    s = f"{v:+.2f}%"
    color = "green" if v >= color_threshold else "red"
    return Text(s, style=color)


def fmt_sharpe(v) -> Text:
    if v is None:
        return Text("—", style="dim")
    s = f"{v:+.3f}"
    color = "green" if v >= 0.4 else ("yellow" if v >= 0 else "red")
    return Text(s, style=color)


def build_layout(state: dict, cfg: dict, events: list[dict]) -> Layout:
    now = datetime.now(timezone.utc)
    console_width = 110

    # ── Header ───────────────────────────────────────────────────────────────
    paper = cfg.get("paper_mode", True)
    mode  = Text("PAPER MODE", style="bold yellow") if paper else Text("LIVE", style="bold red")
    exchange = cfg.get("exchange", "?").upper()
    header = Text(f"  MOMENTUM STRATEGY  |  {exchange}  |  ", style="bold white")
    header.append_text(mode)
    header.append(f"  |  {now.strftime('%Y-%m-%d %H:%M:%S')} UTC", style="dim")

    # ── Portfolio summary ─────────────────────────────────────────────────────
    # Paper mode: value = cash (paper_usdt) + mark-to-market of holdings.
    # We don't have live prices in the dashboard, so use the last logged value
    # from pnl_history, falling back to paper_usdt, then capital_usdt.
    pnl_hist_raw = state.get("pnl_history", [])
    if pnl_hist_raw:
        pf_val = float(pnl_hist_raw[-1]["value"])
    else:
        pf_val = float(state.get("paper_usdt") or state.get("portfolio_value") or
                       cfg.get("capital_usdt", 500))
    inception = float(state.get("inception_value") or cfg.get("capital_usdt", 500))
    peak      = float(state.get("peak_value") or pf_val)
    pnl_pct   = (pf_val / inception - 1) * 100 if inception else 0
    dd_pct    = (pf_val / peak - 1) * 100 if peak else 0
    rebal_cnt = state.get("rebal_count", 0)

    # Rolling Sharpe from pnl_history
    pnl_hist = state.get("pnl_history", [])
    rs = None
    if len(pnl_hist) >= 10:
        import numpy as np
        vals = [p["value"] for p in pnl_hist[-168:]]
        rets = np.diff(vals) / np.array(vals[:-1])
        if rets.std() > 0:
            rs = round(rets.mean() / rets.std() * (8760 ** 0.5), 3)

    # Last rebalance & next
    last_rebal_ts = state.get("last_rebal_ts")
    rebal_interval = cfg.get("rebalance_interval_hours", 168)
    if last_rebal_ts:
        last_dt  = datetime.fromisoformat(last_rebal_ts)
        next_dt  = last_dt + timedelta(hours=rebal_interval)
        hrs_to   = (next_dt - now).total_seconds() / 3600
        next_str = next_dt.strftime("%Y-%m-%d %H:%M UTC")
        due_str  = f"in {hrs_to:.1f}h" if hrs_to > 0 else "OVERDUE"
    else:
        next_str = "pending first rebalance"
        due_str  = "—"

    pf_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    pf_table.add_column("Key",   style="dim", width=26)
    pf_table.add_column("Value", width=22)
    pf_table.add_column("Key2",  style="dim", width=26)
    pf_table.add_column("Value2", width=22)

    pf_val_fmt = Text(f"${pf_val:,.2f}", style="bold cyan")
    pf_table.add_row("Portfolio value",  pf_val_fmt,
                     "Max drawdown",     fmt_pct(dd_pct))
    pf_table.add_row("P&L since inception", fmt_pct(pnl_pct),
                     "Rolling 7d Sharpe",   fmt_sharpe(rs))
    pf_table.add_row("Rebalances done",  str(rebal_cnt),
                     "Next rebalance",   Text(f"{next_str} ({due_str})", style="cyan"))

    # ── Circuit breaker ───────────────────────────────────────────────────────
    cb = state.get("circuit_breaker")
    cb_panel = None
    if cb:
        cb_panel = Panel(
            Text(f"⚠  CIRCUIT BREAKER ACTIVE\n{cb}\n\nTo resume: clear 'circuit_breaker' in state.json",
                 style="bold red"),
            border_style="red", title="HALT"
        )

    # ── Positions table ───────────────────────────────────────────────────────
    pos_table = Table(title="Current Positions", box=box.SIMPLE_HEAVY,
                      show_header=True, header_style="bold blue")
    pos_table.add_column("Asset",  width=8)
    pos_table.add_column("Qty",    width=14, justify="right")
    pos_table.add_column("Weight", width=10, justify="right")
    pos_table.add_column("Value (USDT)", width=16, justify="right")
    pos_table.add_column("Status", width=10)

    positions = state.get("positions", {})
    assets    = cfg.get("assets", [])
    for asset in assets:
        qty    = positions.get(asset, 0)
        val    = 0  # we don't store last price in state; show qty only
        weight = val / pf_val * 100 if pf_val else 0
        status = Text("HOLD", style="green") if qty > 0 else Text("CASH", style="dim")
        qty_s  = f"{qty:.6f}" if qty > 0 else "—"
        val_s  = f"${val:,.2f}" if val > 0 else "—"
        w_s    = f"{weight:.1f}%" if weight > 0 else "—"
        pos_table.add_row(asset, qty_s, w_s, val_s, status)

    # ── Recent log events ─────────────────────────────────────────────────────
    log_table = Table(title="Recent Events", box=box.SIMPLE,
                      show_header=True, header_style="bold")
    log_table.add_column("Time",  width=10, style="dim")
    log_table.add_column("Event", width=22)
    log_table.add_column("Detail", width=60)

    event_styles = {
        "rebalance_complete":   "green",
        "rebalance_start":      "cyan",
        "circuit_breaker_triggered": "bold red",
        "order_filled":         "blue",
        "order_error":          "red",
        "network_error":        "yellow",
        "engine_start":         "white",
        "hourly_check":         "dim",
    }

    shown = [e for e in events if e.get("event") != "hourly_check"][-12:]
    for ev in reversed(shown):
        ts_str = ev.get("ts", "")[:19].replace("T", " ")[11:]
        evname = ev.get("event", "?")
        style  = event_styles.get(evname, "white")

        # Build detail string
        skip   = {"ts", "event"}
        detail = "  ".join(f"{k}={v}" for k, v in ev.items()
                           if k not in skip and not isinstance(v, (dict, list)))
        log_table.add_row(ts_str, Text(evname, style=style), Text(detail[:80], style="dim"))

    # ── Assemble layout ───────────────────────────────────────────────────────
    layout = Layout()
    layout.split_column(
        Layout(Panel(header, box=box.HEAVY_HEAD, border_style="blue"), size=3),
        Layout(Panel(pf_table, title="Portfolio", border_style="cyan"), size=8),
        Layout(name="mid"),
        Layout(Panel(log_table, border_style="dim"), size=18),
        Layout(Text(f"  Refreshing every {REFRESH_SECS}s — Ctrl+C to quit",
                    style="dim italic"), size=1),
    )
    layout["mid"].split_row(
        Layout(Panel(pos_table, border_style="blue"), ratio=1),
        Layout(cb_panel or Panel(
            Text("✓ No circuit breakers active", style="green"),
            title="Circuit Breakers", border_style="green"), ratio=1),
    )
    return layout


def main():
    console = Console()
    console.clear()

    with Live(console=console, refresh_per_second=0.5, screen=True) as live:
        while True:
            try:
                state  = load_state()
                cfg    = load_config()
                events = load_recent_logs()
                layout = build_layout(state, cfg, events)
                live.update(layout)
                time.sleep(REFRESH_SECS)
            except KeyboardInterrupt:
                break
            except Exception as e:
                live.update(Panel(Text(f"Dashboard error: {e}", style="red")))
                time.sleep(5)

    console.print("Dashboard closed.")


if __name__ == "__main__":
    main()
