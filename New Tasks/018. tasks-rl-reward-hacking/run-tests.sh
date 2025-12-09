#!/bin/bash
set -e

# 1. Run the agent's training loop (using their modified env)
# This will generate the Q-Table
python3 /app/src/train.py

# 2. Run the validator
python3 -m pytest -v tests/test_outputs.py