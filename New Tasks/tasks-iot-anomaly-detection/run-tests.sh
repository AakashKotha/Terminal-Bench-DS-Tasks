#!/bin/bash
set -e

echo "Starting test run..."

# Install uv (Python package manager)
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source /root/.cargo/env

# Install pytest and pandas (pandas is used by the test script)
echo "Installing pytest and pandas..."
uv pip install pytest pandas

# Run pytest
echo "Running pytest..."
uv run pytest

echo "Test run finished."