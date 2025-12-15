#!/bin/bash
set -e

# 1. Run the agent's script. 
# We expect the agent to have modified this file to fix the logic.
# Running it generates the /app/output/recommendations.json file.
python3 /app/src/broken_search.py

# 2. Run the rigorous validation
python3 -m pytest -v tests/test_outputs.py