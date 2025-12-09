#!/bin/bash
set -e

# Run pytest on the test definitions
# -v: verbose output
# -x: stop on first failure
python3 -m pytest -v -x tests/test_outputs.py