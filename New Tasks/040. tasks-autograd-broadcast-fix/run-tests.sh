#!/bin/bash
# Run the pytest suite
# The tests are mounted at /tests
# We assume the agent has modified /app/nanograd.py in place.

# Run pytest on the mounted tests directory
python3 -m pytest /tests/test_outputs.py -v