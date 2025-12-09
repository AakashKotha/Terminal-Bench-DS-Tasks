import os
import pytest

PRED_PATH = "/app/output/predictions.txt"
TRUTH_PATH = "/app/tests/hidden/ground_truth.txt"

def test_file_exists():
    assert os.path.exists(PRED_PATH), f"Output file {PRED_PATH} not found. Did the script run?"

def test_prediction_accuracy():
    # Load predictions
    with open(PRED_PATH, 'r') as f:
        preds = f.read().strip()
        
    # Load truth
    with open(TRUTH_PATH, 'r') as f:
        truth = f.read().strip()
        
    # Check length
    assert len(preds) == len(truth), \
        f"Prediction length mismatch. Expected {len(truth)}, got {len(preds)}."
    
    # Calculate accuracy
    matches = sum(1 for p, t in zip(preds, truth) if p == t)
    accuracy = matches / len(truth)
    
    print(f"DEBUG: Agent Accuracy: {accuracy:.4f}")
    
    # Threshold:
    # Random guessing or 'All Introns' would give ~50% (since states are roughly balanced in generation).
    # A working Viterbi on this clean data should get > 95%.
    assert accuracy > 0.90, \
        f"FAILED: Accuracy is too low ({accuracy:.2%}).\n" \
        "This indicates the Viterbi algorithm failed, likely due to floating point underflow.\n" \
        "Did you convert the multiplication logic to Log-Space addition?"