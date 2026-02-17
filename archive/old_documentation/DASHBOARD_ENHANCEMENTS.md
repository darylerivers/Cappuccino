# Dashboard Enhancements - Interactive Features Guide

## Overview
The Cappuccino Dashboard has been enhanced with interactive controls, quick navigation, system health monitoring, and real-time updates.

## New Features

### 1. **Quick Navigation** ⌨️
- **Number Keys (0-8)**: Jump directly to any page
  - `0` - Main Dashboard
  - `1` - Ensemble Voting
  - `2` - Portfolio History
  - `3` - Training Statistics
  - `4` - Trade History
  - `5` - Macro Indicators
  - `6` - News Monitor
  - `7` - Model Arena
  - `8` - System Health (NEW!)

- **Vim-style Navigation**:
  - `h` / `←` - Previous page
  - `l` / `→` - Next page

### 2. **Help Overlay** ❓
- Press `?` to show/hide keyboard shortcuts help
- Displays all available commands and page shortcuts
- Press any key to close

### 3. **System Health Page** (Page 8) 🏥
Comprehensive system monitoring dashboard showing:

#### Services Status
- Training Workers (with worker count)
- Paper Trader
- Arena Runner
- Auto Model Deployer
- Autonomous Advisor
- System Watchdog

Each service shows:
- Status (RUNNING/STOPPED)
- Process ID (PID)
- Additional info

#### System Resources
- **Disk Usage**: Used/Total space and percentage
- **Memory**: Used/Total RAM
- **GPU**: Utilization %, memory usage, temperature (per GPU)

#### Recent Alerts
- Last 5 notifications/alerts
- Color-coded by severity (info/warning/error)
- Timestamp for each alert

#### Quick Actions
- `[s]` Open Service Control Menu
- `[r]` Refresh Status
- `[?]` Show Help

### 4. **Service Control Menu** 🎛️
Interactive menu to start/stop services directly from the dashboard:

**Access**: Press `s` from any page

**Controls**:
- `↑/↓` - Select service
- `Enter` - Toggle service (start if stopped, stop if running)
- `ESC` or `q` - Close menu

**Available Services**:
1. Training Workers
2. Paper Trader
3. Arena Runner

Each shows current status and what action will be performed.

### 5. **Auto-Refresh Control** 🔄
- Press `a` to toggle auto-refresh on/off
- Status shown in navigation bar: `Auto[ON]` or `Auto[OFF]`
- Green when ON, red when OFF
- Manual refresh always available with `r` key

### 6. **Notification System** 🔔
Automatic background monitoring that checks every 30 seconds for:
- **Low disk space** (< 20GB remaining)
- **Paper trader heartbeat** (stale if > 10 minutes old)
- **Service starts/stops** (from service control menu)

Notifications appear in:
- System Health page (last 5 alerts)
- Kept in memory (last 20 notifications)

## Updated Navigation Bar
Consistent across all pages showing:
```
Page X/9 | 0-8 Jump | ←/→ Nav | r Refresh | a Auto[ON] | s Services | ? Help | q Quit
```

## Keyboard Reference

### Navigation
| Key | Action |
|-----|--------|
| `←` or `h` | Previous page |
| `→` or `l` | Next page |
| `0-8` | Jump to specific page |
| `q` | Quit dashboard |

### Controls
| Key | Action |
|-----|--------|
| `r` | Manual refresh |
| `a` | Toggle auto-refresh |
| `s` | Open service control menu |
| `?` | Show/hide help overlay |
| `ESC` | Close overlays |

### Service Menu (when open)
| Key | Action |
|-----|--------|
| `↑` | Select previous service |
| `↓` | Select next service |
| `Enter` | Toggle selected service |
| `q` or `ESC` | Close menu |

## Usage Examples

### Quick Service Management
1. Press `8` to jump to System Health page
2. Check which services are running
3. Press `s` to open service control menu
4. Use `↑/↓` to select a service
5. Press `Enter` to toggle it on/off
6. Press `ESC` to return to dashboard

### Monitoring While Working
1. Start dashboard: `python3 dashboard.py`
2. Press `a` to disable auto-refresh (saves resources)
3. Navigate with number keys `0-8` to check different aspects
4. Press `r` when you want to manually refresh
5. Check `8` (System Health) periodically for alerts

### Quick Status Check
1. Press `8` to jump directly to System Health
2. Scan services status (all green = good)
3. Check GPU utilization
4. Review recent alerts
5. Navigate back with `0` for main dashboard

## Technical Details

### State Management
- Dashboard maintains UI state for:
  - Current page (0-8)
  - Help overlay visibility
  - Service menu visibility
  - Auto-refresh toggle status
  - Selected service in menu
  - Notification queue (last 20)

### Performance
- Auto-refresh respects toggle state (no unnecessary renders)
- Background notification checks every 30 seconds
- Tiburtina data pre-warmed in background thread
- Service status checks are non-blocking

### Error Handling
- Graceful degradation if services unavailable
- All operations wrapped in try-except
- Errors logged to notifications
- Dashboard continues running on errors

## Files Modified
- `dashboard.py` - All enhancements integrated
  - Added 9th page (System Health)
  - Added help overlay
  - Added service control menu
  - Enhanced keyboard handling
  - Added notification system
  - Added auto-refresh toggle

## Requirements
No new dependencies - uses existing subprocess, os, Path modules for system monitoring.

## Tips
- Use `?` to see all shortcuts anytime
- Use number keys for fastest navigation
- Check System Health (page 8) first thing for quick status
- Service menu is safer than manual shell commands
- Auto-refresh OFF saves CPU when you don't need live updates
