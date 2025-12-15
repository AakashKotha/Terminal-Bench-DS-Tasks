#!/bin/bash
set -e

# Setup Python environment
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

uv venv .venv
source .venv/bin/activate

# Install dependencies
# CPU torch is sufficient
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install pytest numpy

# Add task-deps to python path
export PYTHONPATH=$PYTHONPATH:$(pwd)/task-deps

# Run tests
pytest tests/test_outputs.py