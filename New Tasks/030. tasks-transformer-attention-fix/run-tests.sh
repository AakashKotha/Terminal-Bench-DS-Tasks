#!/bin/bash
set -e

# Setup output directory
mkdir -p /app/output

# 1. Run the training script provided in the environment.
# This serves as the primary verification. If the user fixed the bug, this exits with 0.
# If the model fails to learn (acc < 0.5), the python script exits with 1.
echo "Running Training Script to verify convergence..."
python3 /app/task-deps/train.py

# 2. Run the pytest suite for sanity checks and regression testing
echo "Running Pytest verification..."
python3 -m pytest -v /app/tests/test_outputs.py