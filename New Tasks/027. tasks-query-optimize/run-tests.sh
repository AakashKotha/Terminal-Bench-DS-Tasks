#!/bin/bash
set -e

# If the agent created a create_indices.sql script, run it first to apply changes
if [ -f /app/create_indices.sql ]; then
    echo "Applying indices from /app/create_indices.sql..."
    sqlite3 /app/db/analytics.db < /app/create_indices.sql
fi

python3 -m pytest -v tests/test_outputs.py