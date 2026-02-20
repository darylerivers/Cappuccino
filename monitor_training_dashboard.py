#!/usr/bin/env python3
"""
Training Dashboard - Comprehensive Real-time Monitoring

Auto-detects active Optuna study via .current_study file or running workers.
All trial data pulled directly from SQLite - no log parsing fragility.

Shows:
- Study progress and best trials
- Sharpe distribution (percentiles, thresholds)
- Trial velocity and convergence trend
- Recent trials with duration and key params
- Best trial full parameter set
- Worker status and resource usage
- GPU utilization (ROCm/AMD)
"""

import sqlite3
import subprocess
import sys
import re
from utils.study_config import get_current_study
from pathlib import Path
from datetime import datetime, timedelta
import json


class Colors:
    HEADER = '\033[95m'
    BLUE   = '\033[94m'
    CYAN   = '\033[96m'
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    END    = '\033[0m'
    BOLD   = '\033[1m'
    DIM    = '\033[2m'


def C(text, color):
    return f"{color}{text}{Colors.END}"


def run_cmd(cmd, timeout=5):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip()
    except Exception:
        return ""


# ── Database helpers ──────────────────────────────────────────────────────────

def _connect(db_path):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_study_id(cursor, study_name):
    cursor.execute("SELECT study_id FROM studies WHERE study_name = ?", (study_name,))
    row = cursor.fetchone()
    return row[0] if row else None


def get_optuna_stats(db_path, study_name):
    """Return comprehensive study statistics."""
    try:
        conn = _connect(db_path)
        cur = conn.cursor()
        study_id = get_study_id(cur, study_name)
        if study_id is None:
            conn.close()
            return None

        # State counts
        cur.execute("""
            SELECT state, COUNT(*) FROM trials WHERE study_id=? GROUP BY state
        """, (study_id,))
        state_counts = {row[0]: row[1] for row in cur.fetchall()}

        # Value stats over all COMPLETE trials
        cur.execute("""
            SELECT tv.value
            FROM trials t JOIN trial_values tv ON t.trial_id=tv.trial_id
            WHERE t.study_id=? AND t.state='COMPLETE' AND tv.value IS NOT NULL
            ORDER BY tv.value
        """, (study_id,))
        values = [row[0] for row in cur.fetchall()]

        # Best 10 trials with params
        cur.execute("""
            SELECT t.number, tv.value, t.trial_id,
                   t.datetime_start, t.datetime_complete
            FROM trials t JOIN trial_values tv ON t.trial_id=tv.trial_id
            WHERE t.study_id=? AND t.state='COMPLETE' AND tv.value IS NOT NULL
            ORDER BY tv.value DESC LIMIT 10
        """, (study_id,))
        best_rows = cur.fetchall()

        best_trials = []
        for row in best_rows:
            cur.execute("""
                SELECT param_name, param_value FROM trial_params WHERE trial_id=?
            """, (row['trial_id'],))
            params = {r[0]: r[1] for r in cur.fetchall()}
            dur = None
            if row['datetime_start'] and row['datetime_complete']:
                try:
                    t0 = datetime.fromisoformat(row['datetime_start'])
                    t1 = datetime.fromisoformat(row['datetime_complete'])
                    dur = int((t1 - t0).total_seconds())
                except Exception:
                    pass
            best_trials.append({
                'number': row['number'],
                'value': row['value'],
                'params': params,
                'duration': dur,
            })

        # Recent 15 COMPLETE trials (by completion time)
        cur.execute("""
            SELECT t.number, tv.value, t.datetime_start, t.datetime_complete, t.trial_id
            FROM trials t JOIN trial_values tv ON t.trial_id=tv.trial_id
            WHERE t.study_id=? AND t.state='COMPLETE' AND tv.value IS NOT NULL
            ORDER BY t.datetime_complete DESC LIMIT 15
        """, (study_id,))
        recent_rows = cur.fetchall()

        recent_trials = []
        for row in recent_rows:
            cur.execute("""
                SELECT param_name, param_value FROM trial_params
                WHERE trial_id=? AND param_name IN
                  ('learning_rate','net_dimension','lookback','lookback_period',
                   'ppo_epochs','batch_size','gamma','n_envs','timeframe')
            """, (row['trial_id'],))
            params = {r[0]: r[1] for r in cur.fetchall()}
            dur = None
            if row['datetime_start'] and row['datetime_complete']:
                try:
                    t0 = datetime.fromisoformat(row['datetime_start'])
                    t1 = datetime.fromisoformat(row['datetime_complete'])
                    dur = int((t1 - t0).total_seconds())
                except Exception:
                    pass
            recent_trials.append({
                'number': row['number'],
                'value': row['value'],
                'params': params,
                'duration': dur,
                'completed_at': row['datetime_complete'],
            })

        # Trial velocity: completed in last 1h, 6h, 24h
        now = datetime.now()
        velocities = {}
        for hours, label in [(1, '1h'), (6, '6h'), (24, '24h')]:
            since = (now - timedelta(hours=hours)).isoformat()
            cur.execute("""
                SELECT COUNT(*) FROM trials t
                WHERE t.study_id=? AND t.state='COMPLETE'
                  AND t.datetime_complete >= ?
            """, (study_id, since))
            velocities[label] = cur.fetchone()[0]

        # Trend: rolling 10-trial sharpe averages (3 windows)
        cur.execute("""
            SELECT tv.value
            FROM trials t JOIN trial_values tv ON t.trial_id=tv.trial_id
            WHERE t.study_id=? AND t.state='COMPLETE' AND tv.value IS NOT NULL
            ORDER BY t.datetime_complete DESC LIMIT 30
        """, (study_id,))
        trend_vals = [row[0] for row in cur.fetchall()]

        conn.close()

        def pct(lst, p):
            if not lst:
                return None
            idx = int(len(lst) * p / 100)
            idx = min(idx, len(lst) - 1)
            return lst[idx]

        total = sum(state_counts.values())
        complete = state_counts.get('COMPLETE', 0)
        pruned = state_counts.get('PRUNED', 0)
        failed = state_counts.get('FAIL', 0)
        running = state_counts.get('RUNNING', 0)

        return {
            'total': total,
            'complete': complete,
            'running': running,
            'failed': failed,
            'pruned': pruned,
            'prune_rate': pruned / max(total, 1) * 100,
            'complete_rate': complete / max(total - running, 1) * 100,
            'best_value': values[-1] if values else None,
            'avg_value': sum(values) / len(values) if values else None,
            'p25': pct(values, 25),
            'p50': pct(values, 50),
            'p75': pct(values, 75),
            'p90': pct(values, 90),
            'p95': pct(values, 95),
            'above_005': sum(1 for v in values if v > 0.05),
            'above_010': sum(1 for v in values if v > 0.10),
            'above_015': sum(1 for v in values if v > 0.15),
            'n_values': len(values),
            'best_trials': best_trials,
            'recent_trials': recent_trials,
            'velocities': velocities,
            'trend_vals': trend_vals,
        }
    except Exception as e:
        print(f"DB error: {e}")
        return None


# ── GPU (ROCm/AMD) ────────────────────────────────────────────────────────────

def get_gpu_status():
    """Get GPU stats via rocm-smi (AMD RX 7900 GRE)."""
    util_out = run_cmd("rocm-smi --showuse 2>/dev/null")
    temp_out = run_cmd("rocm-smi --showtemp 2>/dev/null")
    mem_out  = run_cmd("rocm-smi --showmeminfo vram 2>/dev/null")

    gpu = {}
    m = re.search(r'GPU use \(%\): (\d+)', util_out)
    gpu['util'] = int(m.group(1)) if m else None

    # Use junction temp as the hottest sensor
    m = re.search(r'Sensor junction\) \(C\): ([\d.]+)', temp_out)
    if not m:
        m = re.search(r'Sensor edge\) \(C\): ([\d.]+)', temp_out)
    gpu['temp'] = float(m.group(1)) if m else None

    m_total = re.search(r'VRAM Total Memory \(B\): (\d+)', mem_out)
    m_used  = re.search(r'VRAM Total Used Memory \(B\): (\d+)', mem_out)
    if m_total and m_used:
        gpu['mem_total_mb'] = int(m_total.group(1)) // (1024 * 1024)
        gpu['mem_used_mb']  = int(m_used.group(1))  // (1024 * 1024)
    else:
        gpu['mem_total_mb'] = gpu['mem_used_mb'] = None

    return gpu if any(v is not None for v in gpu.values()) else None


# ── Workers ───────────────────────────────────────────────────────────────────

def get_worker_status():
    output = run_cmd("ps aux | grep '1_optimize_unified.py' | grep -v grep")
    workers = []
    for line in output.split('\n'):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 11:
            workers.append({
                'pid': parts[1],
                'cpu': parts[2],
                'mem_pct': parts[3],
                'mem_mb': int(float(parts[5]) / 1024),
                'time': parts[9],
            })
    return workers


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_dur(seconds):
    if seconds is None:
        return '  --  '
    if seconds < 60:
        return f'{seconds}s'
    m, s = divmod(seconds, 60)
    if m < 60:
        return f'{m}m{s:02d}s'
    h, m = divmod(m, 60)
    return f'{h}h{m:02d}m'


def sharpe_color(v):
    if v is None:
        return Colors.DIM
    if v >= 0.15:
        return Colors.GREEN
    if v >= 0.05:
        return Colors.CYAN
    if v >= 0.0:
        return Colors.YELLOW
    return Colors.RED


def bar(fraction, width=20, fill='█', empty='░'):
    n = max(0, min(width, int(fraction * width)))
    return fill * n + empty * (width - n)


def trend_arrow(vals):
    """Compare mean of last 5 vs prior 5 trials, return arrow string."""
    if len(vals) < 6:
        return C('  --  ', Colors.DIM)
    recent  = sum(vals[:5])  / 5
    earlier = sum(vals[5:10]) / min(len(vals[5:10]), 5)
    diff = recent - earlier
    if diff > 0.005:
        return C(f'+{diff:.4f} ▲', Colors.GREEN)
    if diff < -0.005:
        return C(f'{diff:.4f} ▼', Colors.RED)
    return C(f'{diff:+.4f} ─', Colors.YELLOW)


# ── Print sections ────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{C('━'*80, Colors.CYAN)}")
    print(C(f'  {title}', Colors.CYAN + Colors.BOLD))
    print(C('━'*80, Colors.CYAN))


def print_dashboard(study_name, db_path, log_dir):
    W = 80

    print(f"\n{C('═'*W, Colors.HEADER)}")
    print(C(f'  CAPPUCCINO TRAINING DASHBOARD', Colors.HEADER + Colors.BOLD))
    print(C(f'  Study: {study_name}', Colors.HEADER))
    print(C(f'  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', Colors.DIM))
    print(C('═'*W, Colors.HEADER))

    stats = get_optuna_stats(db_path, study_name)

    # ── PROGRESS ──────────────────────────────────────────────────────────────
    section('STUDY PROGRESS')
    if stats:
        total    = stats['total']
        complete = stats['complete']
        pruned   = stats['pruned']
        failed   = stats['failed']
        running  = stats['running']

        bfill = bar(complete / max(total, 1), width=30)
        pfill = bar(pruned / max(total, 1), width=30)

        print(f"  Trials         {C(str(total), Colors.BOLD)}")
        print(f"  ├─ Complete  {C(str(complete).rjust(6), Colors.GREEN)}  [{C(bfill, Colors.GREEN)}]  {stats['complete_rate']:.1f}% of finished")
        print(f"  ├─ Pruned    {C(str(pruned).rjust(6), Colors.DIM)}  [{C(pfill, Colors.DIM)}]  {stats['prune_rate']:.1f}%")
        print(f"  ├─ Failed    {C(str(failed).rjust(6), Colors.RED)}")
        print(f"  └─ Running   {C(str(running).rjust(6), Colors.YELLOW)}")

        print()
        v = stats['velocities']
        print(f"  Trial velocity  "
              f"1h: {C(str(v['1h']).rjust(3), Colors.CYAN)}  "
              f"6h: {C(str(v['6h']).rjust(3), Colors.CYAN)}  "
              f"24h: {C(str(v['24h']).rjust(4), Colors.CYAN)}")

        trend = trend_arrow(stats['trend_vals'])
        print(f"  10-trial trend  {trend}  (last 5 vs prior 5 avg Sharpe)")
    else:
        print(f"  {C('No study data available', Colors.YELLOW)}")

    # ── SHARPE DISTRIBUTION ───────────────────────────────────────────────────
    section('SHARPE DISTRIBUTION  (COMPLETE trials)')
    if stats and stats['n_values'] > 0:
        best = stats['best_value']
        p95  = stats['p95']
        p90  = stats['p90']
        p75  = stats['p75']
        p50  = stats['p50']
        p25  = stats['p25']
        avg  = stats['avg_value']

        def sv(v):
            if v is None:
                return C('  n/a  ', Colors.DIM)
            return C(f'{v:+.6f}', sharpe_color(v))

        print(f"  Best            {sv(best)}")
        print(f"  p95             {sv(p95)}")
        print(f"  p90             {sv(p90)}")
        print(f"  p75             {sv(p75)}")
        print(f"  Median (p50)    {sv(p50)}")
        print(f"  p25             {sv(p25)}")
        print(f"  Mean            {sv(avg)}")
        print()
        n = stats['n_values']
        print(f"  Above 0.15      {C(str(stats['above_015']).rjust(4), Colors.GREEN)}  / {n}  ({stats['above_015']/max(n,1)*100:.1f}%)")
        print(f"  Above 0.10      {C(str(stats['above_010']).rjust(4), Colors.CYAN)}  / {n}  ({stats['above_010']/max(n,1)*100:.1f}%)")
        print(f"  Above 0.05      {C(str(stats['above_005']).rjust(4), Colors.CYAN)}  / {n}  ({stats['above_005']/max(n,1)*100:.1f}%)")
    else:
        print(f"  {C('No complete trials yet', Colors.YELLOW)}")

    # ── BEST TRIALS ───────────────────────────────────────────────────────────
    section('TOP 10 TRIALS')
    if stats and stats['best_trials']:
        hdr = f"  {'#':>5}  {'Sharpe':>10}  {'Dur':>7}  Key params"
        print(hdr)
        print(f"  {'─'*74}")
        for t in stats['best_trials']:
            v   = t['value']
            dur = fmt_dur(t['duration'])
            p   = t['params']
            lr  = float(p.get('learning_rate', 0) or 0)
            nd  = int(float(p.get('net_dimension', 0) or 0))
            lb  = int(float(p.get('lookback', p.get('lookback_period', 0)) or 0))
            ep  = int(float(p.get('ppo_epochs', 0) or 0))
            g   = float(p.get('gamma', 0) or 0)
            kp  = f"LR:{lr:.1e}  NetDim:{nd:4d}  LB:{lb:2d}  PPOep:{ep}  γ:{g:.3f}"
            print(f"  {t['number']:>5}  {C(f'{v:+.6f}', sharpe_color(v)):>10}  {C(dur, Colors.DIM):>7}  {kp}")
    else:
        print(f"  {C('No complete trials', Colors.YELLOW)}")

    # ── RECENT TRIALS ─────────────────────────────────────────────────────────
    section('RECENT 15 TRIALS  (latest first)')
    if stats and stats['recent_trials']:
        hdr = f"  {'#':>5}  {'Sharpe':>10}  {'Dur':>7}  Completed at          Key params"
        print(hdr)
        print(f"  {'─'*74}")
        for t in stats['recent_trials']:
            v   = t['value']
            dur = fmt_dur(t['duration'])
            ts  = (t['completed_at'] or '')[:16]
            p   = t['params']
            lr  = float(p.get('learning_rate', 0) or 0)
            nd  = int(float(p.get('net_dimension', 0) or 0))
            lb  = int(float(p.get('lookback', p.get('lookback_period', 0)) or 0))
            kp  = f"LR:{lr:.1e}  NetDim:{nd:4d}  LB:{lb}"
            print(f"  {t['number']:>5}  {C(f'{v:+.6f}', sharpe_color(v)):>10}  {C(dur, Colors.DIM):>7}  {C(ts, Colors.DIM):<20}  {kp}")
    else:
        print(f"  {C('No recent trials', Colors.YELLOW)}")

    # ── BEST TRIAL PARAMS ─────────────────────────────────────────────────────
    if stats and stats['best_trials']:
        best = stats['best_trials'][0]
        section(f'BEST TRIAL #{best["number"]}  Sharpe={best["value"]:+.6f}  ({fmt_dur(best["duration"])})')
        p = best['params']
        # Print in sorted groups
        keys = sorted(p.keys())
        col_w = 24
        for i in range(0, len(keys), 3):
            row_keys = keys[i:i+3]
            parts = []
            for k in row_keys:
                raw = p[k]
                try:
                    v = float(raw)
                    if v == int(v):
                        vstr = str(int(v))
                    else:
                        vstr = f'{v:.5g}'
                except (TypeError, ValueError):
                    vstr = str(raw)
                parts.append(f'{k}: {C(vstr, Colors.CYAN)}'.ljust(col_w + len(Colors.CYAN) + len(Colors.END)))
            print('  ' + '  '.join(parts))

    # ── WORKERS ───────────────────────────────────────────────────────────────
    section('TRAINING WORKERS')
    workers = get_worker_status()
    if workers:
        print(f"  Active: {C(str(len(workers)), Colors.GREEN)}")
        print(f"\n  {'PID':>8}  {'CPU%':>6}  {'RAM MB':>8}  {'Runtime':>10}")
        print(f"  {'─'*50}")
        for w in workers:
            cpu = float(w['cpu'])
            cc = Colors.GREEN if cpu > 50 else Colors.YELLOW
            cpu_str = C(w['cpu'] + '%', cc)
            print(f"  {w['pid']:>8}  {cpu_str:>6}  {w['mem_mb']:>8}  {w['time']:>10}")
    else:
        print(f"  {C('No workers running', Colors.RED)}")

    # ── GPU ───────────────────────────────────────────────────────────────────
    section('GPU  (RX 7900 GRE / ROCm)')
    gpu = get_gpu_status()
    if gpu:
        util = gpu['util']
        temp = gpu['temp']
        mu   = gpu['mem_used_mb']
        mt   = gpu['mem_total_mb']

        uc = Colors.GREEN if util is not None and util >= 80 else Colors.YELLOW
        tc = (Colors.RED if temp is not None and temp >= 85
              else Colors.YELLOW if temp is not None and temp >= 70
              else Colors.GREEN)

        util_str = f"{C(str(util)+'%', uc)}" if util is not None else C('n/a', Colors.DIM)
        temp_str = f"{C(str(temp)+'°C', tc)}" if temp is not None else C('n/a', Colors.DIM)

        if mu is not None and mt is not None:
            mem_pct = mu / mt * 100
            mc = Colors.YELLOW if mem_pct > 80 else Colors.GREEN
            mem_str = f"{C(str(mu), mc)} / {mt} MB  ({mem_pct:.1f}%)"
            mem_bar = f"  [{C(bar(mu/mt, width=30), mc)}]"
        else:
            mem_str = C('n/a', Colors.DIM)
            mem_bar = ''

        print(f"  Utilization   {util_str}")
        print(f"  VRAM          {mem_str}")
        print(f"  {mem_bar}")
        print(f"  Temperature   {temp_str}")
    else:
        print(f"  {C('GPU info unavailable', Colors.YELLOW)}")

    print(f"\n{C('═'*W, Colors.HEADER)}\n")


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(description='Cappuccino Training Dashboard')
    parser.add_argument('--study', type=str, default=None,
                        help='Optuna study name (auto-detects from .current_study if omitted)')
    parser.add_argument('--db', type=str, default='databases/optuna_cappuccino.db',
                        help='Optuna database path')
    parser.add_argument('--logs', type=str, default='logs',
                        help='Log directory')
    parser.add_argument('--watch', action='store_true',
                        help='Refresh every 15 seconds')
    parser.add_argument('--interval', type=int, default=15,
                        help='Refresh interval in seconds (default: 15)')
    args = parser.parse_args()

    if args.study is None:
        args.study = get_current_study()
        print(f"Auto-detected study: {args.study}")

    if args.watch:
        try:
            while True:
                print('\033[2J\033[H', end='')  # clear screen
                print_dashboard(args.study, args.db, args.logs)
                print(f"{C(f'Refreshing every {args.interval}s … Ctrl+C to exit', Colors.DIM)}")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\n{C('Dashboard closed.', Colors.GREEN)}\n")
    else:
        print_dashboard(args.study, args.db, args.logs)


if __name__ == '__main__':
    main()
