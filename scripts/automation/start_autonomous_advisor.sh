#!/bin/bash
# Start the autonomous AI advisor

# Load configuration from .env.training
if [ -f ".env.training" ]; then
    source .env.training
    DEFAULT_STUDY="$ACTIVE_STUDY_NAME"
else
    DEFAULT_STUDY="cappuccino_week_20251206"
fi

STUDY_NAME="${1:-$DEFAULT_STUDY}"
MODEL="${2:-qwen2.5-coder:7b}"
ANALYSIS_INTERVAL="${3:-50}"  # Analyze every 50 new trials
CHECK_INTERVAL="${4:-300}"    # Check every 5 minutes
MAX_TEST_TRIALS="${5:-10}"    # Run 10 trials per test config

echo "=========================================="
echo "Starting Autonomous AI Advisor"
echo "=========================================="
echo "Study: $STUDY_NAME"
echo "Model: $MODEL"
echo "Analysis interval: Every $ANALYSIS_INTERVAL new trials"
echo "Check interval: Every $CHECK_INTERVAL seconds"
echo "Test trials per config: $MAX_TEST_TRIALS"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p logs

# Start the advisor in background
python -u ollama_autonomous_advisor.py \
  --study "$STUDY_NAME" \
  --model "$MODEL" \
  --analysis-interval "$ANALYSIS_INTERVAL" \
  --check-interval "$CHECK_INTERVAL" \
  --max-test-trials "$MAX_TEST_TRIALS" \
  --daemon \
  > logs/autonomous_advisor_console.log 2>&1 &

ADVISOR_PID=$!
echo $ADVISOR_PID > logs/autonomous_advisor.pid

echo "Autonomous advisor started with PID: $ADVISOR_PID"
echo "Logs: logs/autonomous_advisor.log"
echo "Console: logs/autonomous_advisor_console.log"
echo ""
echo "Monitor with:"
echo "  tail -f logs/autonomous_advisor.log"
echo ""
echo "Stop with:"
echo "  ./stop_autonomous_advisor.sh"
echo ""
