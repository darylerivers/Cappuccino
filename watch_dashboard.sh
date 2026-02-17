#!/bin/bash
# Convenient wrapper for auto-refresh dashboard

INTERVAL=${1:-30}

echo "Starting detailed dashboard with ${INTERVAL}s refresh..."
echo ""

python scripts/automation/dashboard_detailed.py --refresh "$INTERVAL"
