#!/bin/bash
set -e

# Setup environment
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

uv venv .venv
source .venv/bin/activate

# Install lightweight dependencies
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install pytest

# Setup path
export PYTHONPATH=$PYTHONPATH:/app/task-deps

# Run
pytest tests/test_outputs.py