import pytest
import json
import os
import numpy as np

OUTPUT_PATH = "/app/output/predictions.json"

def test_predictions_file_exists():
    assert os.path.exists(OUTPUT_PATH), "Output file not found. Did you run /app/src/naive_hmm.py?"

def test_long_sequence_success():
    """
    If the agent fixed the code to use log-space, the long sequence
    should have been processed successfully.
    """
    with open(OUTPUT_PATH, 'r') as f:
        res = json.load(f)
        
    assert res.get("success", False) is True, \
        "The model failed to decode the long sequence. Likely still hitting Floating Point Underflow."

def test_accuracy_quality():
    """
    Checks that the logic is actually Viterbi and not just random guessing.
    A properly implemented HMM should easily get >90% on this synthetic data.
    """
    with open(OUTPUT_PATH, 'r') as f:
        res = json.load(f)
        
    acc = res.get("accuracy", 0.0)
    print(f"Verified Accuracy: {acc:.2%}")
    
    assert acc > 0.90, \
        f"Accuracy too low ({acc:.2%}). Check your Log-Space arithmetic logic (summing logs instead of multiplying)."

def test_source_code_inspection():
    """
    Heuristic check: Did they use np.log or math.log?
    Did they replace multiplication with addition?
    """
    with open("/app/src/naive_hmm.py", "r") as f:
        code = f.read()
    
    # We look for log usage
    has_log = "np.log" in code or "math.log" in code
    
    # Ideally, they should initialize V with -infinity or similar low value for log(0)
    # but we can't strictly regex that.
    
    assert has_log, \
        "FAILED: Source code does not appear to use Logarithms. You must use Log-Space arithmetic to solve underflow."