#!/bin/bash
set -e

# Setup python environment
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install pytest pytest-asyncio

# Add task-deps to PYTHONPATH so server can import model
export PYTHONPATH=$PYTHONPATH:/app/task-deps

# Run tests
pytest tests/test_outputs.py