#!/bin/bash
set -e

# 1. Run the agent's processor script (generates output)
python3 /app/src/processor.py

# 2. Run validation
python3 -m pytest -v tests/test_outputs.py