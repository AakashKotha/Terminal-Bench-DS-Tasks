#!/bin/bash
set -e
# Run the pytest suite which executes the agent's code internally
python3 -m pytest -v tests/test_outputs.py