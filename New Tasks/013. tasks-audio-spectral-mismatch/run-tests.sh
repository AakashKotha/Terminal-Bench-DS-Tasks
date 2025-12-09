#!/bin/bash
set -e

# Run the agent's training script first (to generate the model)
# We expect the agent to have fixed it, so it should exit 0 and produce a model.
# If they haven't fixed it, they might assume they just need to run it.
# But for the benchmark validation, we run the pytest which CHECKS the model.

python3 -m pytest -v tests/test_outputs.py