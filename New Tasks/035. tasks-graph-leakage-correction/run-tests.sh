#!/bin/bash
set -e

# Setup Python environment
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

uv venv .venv
source .venv/bin/activate

# Install dependencies (CPU torch is sufficient and faster)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install pytest

# Run tests
export PYTHONPATH=$PYTHONPATH:/app/task-deps
pytest tests/test_outputs.py