#!/bin/bash
set -e

# 1. Run the agent's debug script (which uses their fixed dataset.py)
# This generates the output artifact if successful
python3 /app/src/debug_training.py

# 2. Run rigorous validation
python3 -m pytest -v tests/test_outputs.py