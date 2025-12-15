#!/bin/bash
set -e

# Run agent code
python3 /app/src/control_loop.py

# Run validation
python3 -m pytest -v tests/test_outputs.py