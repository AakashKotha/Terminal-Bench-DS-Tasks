#!/bin/bash
# This script is executed inside the task container to run the tests.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting test run..."

# Install uv (Python package manager)
echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source /root/.cargo/env

# Install test dependencies
echo "Installing pytest, scikit-learn, pandas, joblib, numpy..."
uv pip install pytest scikit-learn pandas joblib numpy

# Run pytest
echo "Running pytest..."
uv run pytest

echo "Test run finished."