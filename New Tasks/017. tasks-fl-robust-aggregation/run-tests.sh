#!/bin/bash
set -e

# Run the agent's server (simulation)
python3 /app/src/fl_server.py

# Run validation
python3 -m pytest -v tests/test_outputs.py