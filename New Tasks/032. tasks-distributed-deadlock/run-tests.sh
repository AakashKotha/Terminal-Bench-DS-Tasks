#!/bin/bash
set -e

# 1. Install dependencies
# We use uv for speed if available, otherwise pip
if command -v uv &> /dev/null; then
    uv venv .venv
    source .venv/bin/activate
    uv pip install torch pytest
else
    pip install torch pytest
fi

# 2. Run the test harness
export PYTHONPATH=$PYTHONPATH:.
pytest tests/test_outputs.py