#!/bin/bash
set -e

# 1. Run the agent's training script first.
# We expect it to SUCCEED (Exit code 0) if they fixed it.
# If it fails/NaNs, the python script itself might exit with error (optional)
# or just save a broken model.
python3 /app/src/train.py

# 2. Run the validation suite
python3 -m pytest -v tests/test_outputs.py