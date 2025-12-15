#!/bin/bash
set -e

# Setup env
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

uv venv .venv
source .venv/bin/activate
uv pip install numpy pytest

# Run tests
export PYTHONPATH=$PYTHONPATH:/app/task-deps
pytest tests/test_outputs.py