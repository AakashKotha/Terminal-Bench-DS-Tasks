#!/bin/bash
set -e

# 1. Run the validation tool (simulating agent checking their work)
# We accept failure here if the agent hasn't run it, but ideally the agent ran it.
# To be safe, we run the agent's code to generate the artifact.
python3 /app/src/validate_alignment.py

# 2. Run the rigorous tests
python3 -m pytest -v tests/test_outputs.py