import os
import json
import pytest
import numpy as np
import sys

OUTPUT_TOKEN = "/app/output/token.json"

def test_audit_passed():
    """
    Checks if the agent ran the audit and passed the sensitivity check.
    """
    assert os.path.exists(OUTPUT_TOKEN), "Audit token not found. Did you run /app/src/audit_privacy.py?"
    
    with open(OUTPUT_TOKEN, "r") as f:
        data = json.load(f)
        
    influence = data.get("influence", 999.0)
    assert influence < 0.2, \
        f"Sensitivity too high ({influence:.4f}). Per-sample clipping logic is likely incorrect."

def test_code_logic_inspection():
    """
    Implicit check:
    We manually import their sanitizer and test it on a specific tensor case
    to ensure they aren't just cheating the audit script.
    """
    sys.path.append("/app/src")
    try:
        from sanitizer import GradientSanitizer
    except ImportError:
        pytest.fail("Could not import GradientSanitizer.")
        
    s = GradientSanitizer(clip_norm=1.0)
    
    # Case: 2 samples. 
    # Sample A: [0.5, 0.5] (Norm ~0.7) -> No Clip
    # Sample B: [100, 100] (Norm ~141) -> Should Clip to 1.0 -> [0.707, 0.707]
    
    # Expected Avg: ([0.5, 0.5] + [0.707, 0.707]) / 2 = [0.6035, 0.6035]
    
    batch = np.array([[0.5, 0.5], [100.0, 100.0]])
    result = s.sanitize_gradients(batch)
    
    expected_val = 0.6035
    actual_val = result[0]
    
    assert np.isclose(actual_val, expected_val, atol=0.1), \
        f"Logic Verification Failed. Expected approx {expected_val}, got {actual_val}. " \
        "Ensure you calculate norms per row (axis=1) and apply clipping before averaging."