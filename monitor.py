"""
Cappuccino System Monitor
Shows: momentum engine status, XGBoost prediction, data pipeline health.

Usage:
    python ~/cappuccino/monitor.py

Refresh: 60s for momentum/pipeline, 600s for XGBoost prediction (expensive).
Ctrl+C to quit.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE         = Path.home() / "cappuccino"
MOMENTUM_DIR = BASE / "momentum"
PIPELINE_DIR = BASE / "data_pipeline"
XGB_DIR      = BASE / "xgboost"

STATE_FILE   = MOMENTUM_DIR / "state.json"
MOM_CFG_FILE = MOMENTUM_DIR / "config.yaml"
COL_LOG      = PIPELINE_DIR / "collector.log"
MOM_LOG_DIR  = MOMENTUM_DIR / "logs"
XGB_METRICS  = XGB_DIR / "results" / "metrics.json"

REFRESH_SECS     = 60
XGB_REFRESH_SECS = 600   # prediction is expensive — refresh every 10 min


# ── Helpers ────────────────────────────────────────────────────────────────────
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def is_alive(name: str) -> tuple[bool, int | None]:
    """Check if a process matching `name` is running. Returns (alive, pid)."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", name], stderr=subprocess.DEVNULL
        ).decode().strip()
        pids = [int(p) for p in out.split() if p.isdigit()]
        return bool(pids), pids[0] if pids else None
    except subprocess.CalledProcessError:
        return False, None


def load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def load_yaml(path: Path) -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def load_recent_logs(n: int = 15) -> list[dict]:
    events = []
    today     = now_utc().strftime("%Y-%m-%d")
    yesterday = (now_utc() - timedelta(days=1)).strftime("%Y-%m-%d")
    for ds in [yesterday, today]:
        lf = MOM_LOG_DIR / f"{ds}.jsonl"
        if lf.exists():
            for line in lf.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return events[-n:]


def last_log_age(log_path: Path) -> str:
    """How long ago was the last line written to a log file."""
    try:
        mtime = log_path.stat().st_mtime
        age   = now_utc().timestamp() - mtime
        if age < 120:
            return f"{int(age)}s ago"
        if age < 7200:
            return f"{age/60:.0f}m ago"
        return f"{age/3600:.1f}h ago"
    except Exception:
        return "unknown"


def fmt_pct(v: float) -> Text:
    s = f"{v:+.2f}%"
    return Text(s, style="green" if v >= 0 else "red")


def run_xgb_prediction() -> list[dict] | None:
    """Run predict.py as subprocess to avoid blocking the monitor loop."""
    try:
        result = subprocess.run(
            [sys.executable, str(XGB_DIR / "predict.py")],
            capture_output=True, text=True, timeout=120,
            cwd=str(XGB_DIR),
        )
        # Parse the table lines from stdout
        lines  = result.stdout.splitlines()
        preds  = []
        in_table = False
        for line in lines:
            if line.startswith("#"):
                in_table = True
            if in_table and line.startswith("#"):
                parts = line.split()
                if len(parts) >= 4:
                    asset   = parts[1]
                    prob    = float(parts[2])
                    conf    = float(parts[3])
                    signal  = "LONG" if prob >= 0.5 else "HOLD"
                    preds.append({
                        "asset": asset, "prob_up": prob,
                        "confidence": conf, "signal": signal,
                    })
        return preds if preds else None
    except Exception:
        return None


def run_xgb_prediction_import() -> list[dict] | None:
    """Import predict directly (faster after first run, uses cached modules)."""
    try:
        sys.path.insert(0, str(XGB_DIR))
        import importlib
        import io
        from contextlib import redirect_stdout

        # Suppress build_features diagnostics output
        with redirect_stdout(io.StringIO()):
            import predict as _pred
            importlib.reload(_pred)   # pick up any file changes
            return _pred.predict()
    except Exception as e:
        return None


# ── Panel builders ─────────────────────────────────────────────────────────────
def build_header() -> Panel:
    now  = now_utc()
    text = Text()
    text.append("  CAPPUCCINO SYSTEM MONITOR", style="bold white")
    text.append(f"  |  {now.strftime('%Y-%m-%d %H:%M:%S')} UTC", style="dim")
    return Panel(text, box=box.HEAVY_HEAD, border_style="blue", height=3)


def build_processes() -> Panel:
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    t.add_column("Process",  width=22)
    t.add_column("Status",   width=12)
    t.add_column("PID",      width=8,  justify="right")
    t.add_column("Notes",    width=30)

    checks = [
        ("engine.py",    "Momentum engine"),
        ("collector.py", "Data pipeline"),
    ]
    for proc, label in checks:
        alive, pid = is_alive(proc)
        status = Text("● RUNNING", style="green bold") if alive else Text("○ DOWN", style="red bold")
        pid_s  = str(pid) if pid else "—"
        t.add_row(label, status, pid_s, proc)

    return Panel(t, title="Processes", border_style="cyan", padding=(0, 1))


def build_momentum(state: dict, cfg: dict, events: list[dict]) -> Panel:
    paper = cfg.get("paper_mode", True)
    mode  = Text("PAPER", style="yellow bold") if paper else Text("LIVE", style="red bold")

    pnl_hist = state.get("pnl_history", [])
    if pnl_hist:
        pf_val = float(pnl_hist[-1]["value"])
    else:
        pf_val = float(state.get("paper_usdt") or cfg.get("capital_usdt", 500))

    inception = float(state.get("inception_value") or cfg.get("capital_usdt", 500))
    peak      = float(state.get("peak_value") or pf_val)
    pnl_pct   = (pf_val / inception - 1) * 100 if inception else 0
    dd_pct    = (pf_val / peak - 1) * 100 if peak else 0

    # Next rebalance
    last_ts  = state.get("last_rebal_ts")
    rebal_h  = cfg.get("rebalance_interval_hours", 168)
    if last_ts:
        last_dt = datetime.fromisoformat(last_ts)
        next_dt = last_dt + timedelta(hours=rebal_h)
        hrs_left = (next_dt - now_utc()).total_seconds() / 3600
        next_s   = f"{next_dt.strftime('%m-%d %H:%M')} UTC  ({hrs_left:.1f}h)"
    else:
        next_s = "pending"

    # Get latest hourly_check event for live prices
    latest_check = next(
        (e for e in reversed(events) if e.get("event") == "hourly_check"), {}
    )
    btc_px    = latest_check.get("btc_price", "?")
    btc_ma    = latest_check.get("btc_ma", "?")
    in_regime = latest_check.get("in_regime", None)
    regime_s  = Text("BULL ▲", style="green") if in_regime else Text("BEAR ▼ (cash)", style="red")
    rankings  = latest_check.get("rankings", {})

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    t.add_column("k", style="dim", width=22)
    t.add_column("v", width=28)
    t.add_column("k2", style="dim", width=18)
    t.add_column("v2", width=20)

    t.add_row("Mode",          mode,
              "Portfolio",     Text(f"${pf_val:,.2f}", style="bold cyan"))
    t.add_row("P&L since start", fmt_pct(pnl_pct),
              "Drawdown",      fmt_pct(dd_pct))
    t.add_row("Regime",        regime_s,
              "Next rebal",    Text(next_s, style="cyan"))
    btc_s = f"${btc_px:,.0f}" if isinstance(btc_px, (int, float)) else str(btc_px)
    ma_s  = f"${btc_ma:,.0f}" if isinstance(btc_ma, (int, float)) else str(btc_ma)
    t.add_row("BTC price",     Text(btc_s, style="white"),
              "BTC MA",        Text(ma_s, style="dim"))

    if rankings:
        sorted_r = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        rank_s   = "  ".join(f"{a}:{v:+.1%}" for a, v in sorted_r)
        t.add_row("24h ranks", Text(rank_s, style="dim"), "", "")

    return Panel(t, title="Momentum Strategy", border_style="cyan", padding=(0, 1))


def build_xgb(preds: list[dict] | None, pred_age_s: float) -> Panel:
    if preds is None:
        return Panel(
            Text("  Prediction unavailable — check XGBoost models", style="dim"),
            title="XGBoost Prediction", border_style="magenta", padding=(0, 1)
        )

    age_s = f"  (computed {pred_age_s:.0f}s ago)" if pred_age_s < 9999 else ""
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    t.add_column("Asset",  width=7)
    t.add_column("P(up)",  width=8, justify="right")
    t.add_column("Conf",   width=8, justify="right")
    t.add_column("Signal", width=10)

    for i, r in enumerate(preds):
        signal = r.get("signal", "?")
        top2   = i < 2 and signal == "LONG"
        sig_t  = Text("★ LONG", style="green bold") if top2 else (
                 Text("LONG",   style="green") if signal == "LONG" else
                 Text("HOLD",   style="dim"))
        t.add_row(r["asset"],
                  f"{r['prob_up']:.4f}",
                  f"{r['confidence']:.4f}",
                  sig_t)

    top_longs = [r["asset"] for i, r in enumerate(preds) if i < 2 and r.get("signal") == "LONG"]
    rec_text  = Text()
    rec_text.append("\n  Recommend: ", style="dim")
    if top_longs:
        rec_text.append(f"LONG {', '.join(top_longs)}", style="green bold")
    else:
        rec_text.append("HOLD all", style="dim")
    rec_text.append(age_s, style="dim italic")

    content = Table.grid()
    content.add_row(t)
    content.add_row(rec_text)

    return Panel(content, title="XGBoost Prediction  (3-fold ensemble)",
                 border_style="magenta", padding=(0, 1))


def load_pipeline_data() -> dict:
    """Query latest derivative signals from collector DB."""
    DB     = PIPELINE_DIR / "binance_features.db"
    ASSETS = ["BTC", "ETH", "DOGE", "ADA", "SOL"]
    SYM    = {a: f"{a}USDT_PERP.A" for a in ASSETS}
    out: dict = {}
    try:
        import sqlite3
        conn   = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        now_ms = int(time.time() * 1000)
        h24    = 86_400_000
        for a, s in SYM.items():
            fr  = conn.execute(
                "SELECT funding_rate, funding_time FROM funding_rates "
                "WHERE symbol=? ORDER BY funding_time DESC LIMIT 1", (s,)
            ).fetchone()
            oi  = conn.execute(
                "SELECT oi_usd, timestamp FROM open_interest "
                "WHERE symbol=? ORDER BY timestamp DESC LIMIT 1", (s,)
            ).fetchone()
            ls  = conn.execute(
                "SELECT long_short_ratio, timestamp FROM ls_ratio "
                "WHERE symbol=? ORDER BY timestamp DESC LIMIT 1", (s,)
            ).fetchone()
            liq = conn.execute(
                "SELECT side, SUM(quantity) FROM liquidations "
                "WHERE symbol=? AND timestamp > ? GROUP BY side",
                (s, now_ms - h24)
            ).fetchall()
            liq_d = {r[0]: r[1] for r in liq}
            ts_candidates = [fr[1] if fr else 0, oi[1] if oi else 0, ls[1] if ls else 0]
            out[a] = {
                "funding":   fr[0]  if fr  else None,
                "oi":        oi[0]  if oi  else None,
                "ls":        ls[0]  if ls  else None,
                "liq_long":  liq_d.get("sell", 0.0),
                "liq_short": liq_d.get("buy",  0.0),
                "ts_ms":     max((v for v in ts_candidates if v), default=0),
            }
        conn.close()
    except Exception:
        pass
    return out


def _fmt_oi(v: float | None) -> str:
    if v is None: return "—"
    if v >= 1e9:  return f"{v/1e9:.2f}B"
    if v >= 1e6:  return f"{v/1e6:.1f}M"
    if v >= 1e3:  return f"{v/1e3:.0f}K"
    return f"{v:.0f}"


def _fmt_liq(v: float) -> str:
    if not v:     return "—"
    if v >= 1e6:  return f"${v/1e6:.1f}M"
    if v >= 1e3:  return f"${v/1e3:.0f}K"
    return f"${v:.0f}"


def build_pipeline() -> Panel:
    log_age    = last_log_age(COL_LOG)
    data       = load_pipeline_data()
    alive, pid = is_alive("collector.py")

    status_t = (Text("● RUNNING", style="green bold") if alive
                else Text("○ DOWN",    style="red bold"))

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold blue",
              padding=(0, 1))
    t.add_column("",        width=5,  style="bold white")
    t.add_column("Funding", width=9,  justify="right")
    t.add_column("OI (raw)", width=9,  justify="right")
    t.add_column("L/S",     width=6,  justify="right")
    t.add_column("Liq Long",  width=9,  justify="right")
    t.add_column("Liq Short", width=9,  justify="right")
    t.add_column("Age",     width=7,  justify="right", style="dim")

    for asset in ["BTC", "ETH", "DOGE", "ADA", "SOL"]:
        d = data.get(asset, {})

        fr = d.get("funding")
        if fr is None:
            fr_t = Text("—", style="dim")
        else:
            fr_t = Text(f"{fr:+.4f}%",
                        style="yellow" if fr > 0.02 else ("cyan" if fr < -0.01 else "white"))

        oi_t = Text(_fmt_oi(d.get("oi")), style="white")

        ls = d.get("ls")
        if ls is None:
            ls_t = Text("—", style="dim")
        else:
            ls_t = Text(f"{ls:.2f}",
                        style="yellow" if ls > 1.5 else ("cyan" if ls < 0.7 else "white"))

        ll   = d.get("liq_long",  0.0)
        lsh  = d.get("liq_short", 0.0)
        ll_t  = Text(_fmt_liq(ll),  style="red"  if ll  > 0 else "dim")
        lsh_t = Text(_fmt_liq(lsh), style="cyan" if lsh > 0 else "dim")

        ts_ms = d.get("ts_ms", 0)
        if ts_ms:
            age_s = now_utc().timestamp() - ts_ms / 1000
            age_t = (f"{age_s/60:.0f}m" if age_s < 7200 else f"{age_s/3600:.1f}h")
        else:
            age_t = "?"

        t.add_row(asset, fr_t, oi_t, ls_t, ll_t, lsh_t, age_t)

    hdr = Table.grid(padding=(0, 1))
    hdr.add_column(); hdr.add_column()
    hdr.add_row(
        Text("Collector ", style="dim") + status_t,
        Text(f"last write: {log_age}", style="dim"),
    )

    grid = Table.grid()
    grid.add_row(hdr)
    grid.add_row(t)

    return Panel(grid, title="Data Pipeline  (Coinalyze · 24h liq)",
                 border_style="blue", padding=(0, 1))


def build_events(events: list[dict]) -> Panel:
    event_styles = {
        "rebalance_complete":        "green",
        "rebalance_start":           "cyan",
        "circuit_breaker_triggered": "bold red",
        "order_filled":              "blue",
        "order_error":               "red",
        "network_error":             "yellow",
        "engine_start":              "white",
        "hourly_check":              "dim",
    }

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    t.add_column("Time",   width=9, style="dim")
    t.add_column("Event",  width=22)
    t.add_column("Detail", width=55)

    shown = [e for e in events if e.get("event") != "hourly_check"][-8:] or events[-6:]
    for ev in reversed(shown):
        ts_s    = ev.get("ts", "")
        time_s  = ts_s[11:19] if len(ts_s) >= 19 else "?"
        evname  = ev.get("event", "?")
        style   = event_styles.get(evname, "white")
        skip    = {"ts", "event"}
        detail  = "  ".join(
            f"{k}={v}" for k, v in ev.items()
            if k not in skip and not isinstance(v, (dict, list))
        )
        t.add_row(time_s, Text(evname, style=style), Text(detail[:70], style="dim"))

    return Panel(t, title="Recent Events (momentum)", border_style="dim", padding=(0, 1))


def build_footer(next_refresh: float, next_xgb: float) -> Text:
    t = Text()
    t.append(f"  Refresh in {next_refresh:.0f}s", style="dim italic")
    t.append(f"  |  XGBoost update in {next_xgb:.0f}s", style="dim italic")
    t.append("  |  Ctrl+C to quit", style="dim italic")
    return t


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    console = Console()

    # Initial XGBoost prediction
    print("Loading XGBoost prediction (first run — please wait ~30s)...")
    xgb_preds   = run_xgb_prediction_import()
    xgb_fetched = time.monotonic()
    last_xgb    = time.monotonic()

    with Live(console=console, refresh_per_second=0.5, screen=True) as live:
        while True:
            try:
                t_now    = time.monotonic()
                pred_age = t_now - xgb_fetched

                # Refresh XGBoost every 10 min (expensive)
                if t_now - last_xgb >= XGB_REFRESH_SECS:
                    xgb_preds   = run_xgb_prediction_import()
                    xgb_fetched = time.monotonic()
                    last_xgb    = time.monotonic()

                state  = load_json(STATE_FILE)
                cfg    = load_yaml(MOM_CFG_FILE)
                events = load_recent_logs()

                next_refresh = REFRESH_SECS - ((time.monotonic() - t_now) % REFRESH_SECS)
                next_xgb     = max(0, XGB_REFRESH_SECS - (time.monotonic() - last_xgb))

                layout = Layout()
                layout.split_column(
                    Layout(build_header(),                          name="header",   size=3),
                    Layout(name="row1",                                              size=11),
                    Layout(name="row2",                                              size=14),
                    Layout(build_events(events),                    name="events",   size=12),
                    Layout(build_footer(next_refresh, next_xgb),   name="footer",   size=1),
                )
                layout["row1"].split_row(
                    Layout(build_processes(), ratio=1),
                    Layout(build_pipeline(),  ratio=2),
                )
                layout["row2"].split_row(
                    Layout(build_momentum(state, cfg, events), ratio=3),
                    Layout(build_xgb(xgb_preds, pred_age),    ratio=2),
                )

                live.update(layout)
                time.sleep(REFRESH_SECS)

            except KeyboardInterrupt:
                break
            except Exception as e:
                live.update(Panel(Text(f"Monitor error: {e}", style="red")))
                time.sleep(5)

    console.print("Monitor closed.")


if __name__ == "__main__":
    main()
