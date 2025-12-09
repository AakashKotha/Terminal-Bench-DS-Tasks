#!/bin/bash
set -e

# 1. Run the audit script (Agent should run this manually, but we enforce it here to generate artifacts)
python3 /app/src/audit_privacy.py || true # Allow fail so pytest catches it properly

# 2. Run Validation
python3 -m pytest -v tests/test_outputs.py