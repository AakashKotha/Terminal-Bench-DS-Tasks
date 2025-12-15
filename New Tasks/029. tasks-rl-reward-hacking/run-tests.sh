#!/bin/bash
set -e

# 1. Run the agent's training script.
# This should retrain the Q-table with the user's fixes.
echo "Running Training Script..."
python3 /app/src/train.py

# 2. Run the validation tests
echo "Running Verification..."
python3 -m pytest -v tests/test_outputs.py