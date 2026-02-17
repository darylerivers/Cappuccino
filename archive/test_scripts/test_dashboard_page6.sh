#!/usr/bin/env bash
# Quick test for dashboard page 6

echo "Testing Dashboard Page 6 Fix..."
echo ""
echo "Starting dashboard (background cache warmup will begin)..."
echo "You should see the main dashboard appear instantly."
echo ""
echo "Instructions:"
echo "1. Press RIGHT ARROW key 5 times to navigate to page 6 (Macro Indicators)"
echo "2. First time might take ~12 seconds (cache warming)"
echo "3. Navigate away and back - should be instant!"
echo "4. Press 'q' to quit"
echo ""
echo "Starting dashboard in 3 seconds..."
sleep 3

python3 dashboard.py
