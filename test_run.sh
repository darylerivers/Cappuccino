#!/bin/bash
# Simple test script to verify background execution works

echo "Script started at $(date)"
echo "Working directory: $(pwd)"
echo "Arguments: $@"

for i in {1..10}; do
    echo "Iteration $i of 10"
    sleep 1
done

echo "Script completed successfully!"
