#!/bin/bash
set -e

# Run the agent's code first to ensure artifacts are fresh
# (Optional, but good for self-contained validation)
# python3 /app/src/train_gnn.py

python3 -m pytest -v tests/test_outputs.py