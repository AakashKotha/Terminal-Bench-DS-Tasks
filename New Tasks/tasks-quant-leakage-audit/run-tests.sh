#!/bin/bash
set -e

# Run the pytest suite
# -v: verbose
# -x: stop on first error
python3 -m pytest -v -x tests/test_outputs.py