#!/bin/bash
# Demo script to test pipeline with trial 965

set -e

cd /opt/user-data/experiment/cappuccino

echo "========================================================================"
echo "PIPELINE DEMO - Testing with Trial 965"
echo "========================================================================"
echo ""

# Enable manual trials in config
echo "Configuring pipeline to use trial 965..."
python << 'PYTHON'
import json

with open('config/pipeline_config.json', 'r') as f:
    config = json.load(f)

config['pipeline']['use_manual_trials'] = True
config['pipeline']['manual_trials'] = [965]

with open('config/pipeline_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("✓ Configuration updated")
PYTHON

# Clear old state
rm -f deployments/pipeline_state.json

echo ""
echo "Running pipeline with trial 965..."
echo ""

# Run pipeline once
python pipeline_orchestrator.py --once

echo ""
echo "========================================================================"
echo "Demo complete! Check logs/pipeline_orchestrator.log for details"
echo "========================================================================"
echo ""

# Restore config
python << 'PYTHON'
import json

with open('config/pipeline_config.json', 'r') as f:
    config = json.load(f)

config['pipeline']['use_manual_trials'] = False

with open('config/pipeline_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("✓ Configuration restored")
PYTHON
