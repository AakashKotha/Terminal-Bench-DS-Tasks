#!/bin/bash
set -e

# Run the agent's solution first to generate the output file
if [ -f "solution.py" ]; then
    echo "Running agent solution..."
    python3 solution.py
else
    echo "Error: solution.py not found"
    exit 1
fi

# Run the validation tests
echo "Running validation tests..."
python3 -m pytest tests/test_outputs.py