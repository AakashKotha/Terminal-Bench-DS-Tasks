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
# Using CPU-only torch to keep image light and fast
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install pytest numpy

# Add task-deps to python path so tests can import the user's modules
export PYTHONPATH=$PYTHONPATH:$(pwd)/task-deps

# Run the test suite
pytest tests/test_outputs.py