import os
import numpy as np
import pytest

MODEL_PATH = "/app/models/global_model.npz"

def test_model_artifact_exists():
    assert os.path.exists(MODEL_PATH), "Model file not found. Did the training complete?"

def test_robustness_against_poison():
    """
    We verify the accuracy stored in the model artifact.
    If the agent used np.mean, accuracy will be < 0.10 (worse than random due to active inversion).
    If the agent used np.median, accuracy will be > 0.90.
    """
    data = np.load(MODEL_PATH)
    acc = float(data["accuracy"])
    
    print(f"Verified Accuracy: {acc:.2%}")
    
    assert acc > 0.90, \
        f"FAILED: Accuracy is too low ({acc:.2%}). " \
        "The malicious client successfully poisoned the global model. " \
        "You must replace 'np.mean' with a robust aggregator like 'np.median' (Coordinate-wise Median)."
