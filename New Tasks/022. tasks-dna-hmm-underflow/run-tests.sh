#!/bin/bash
set -e

# 1. Run the agent's code. 
# Even if it fails (underflow), it writes a JSON file saying "success: false".
# So we allow it to run without `set -e` crashing the script immediately, 
# though Python usually exits 0 unless there's an exception.
python3 /app/src/naive_hmm.py

# 2. Run the rigorous tests
python3 -m pytest -v tests/test_outputs.py