import os
import json
import pytest
import numpy as np

OUTPUT_PATH = "/app/output/results.json"
DATA_PATH = "/app/data/embeddings.npz"

def test_file_exists():
    assert os.path.exists(OUTPUT_PATH), "Results file not found. Did you run the engine?"

def test_recall_performance():
    """
    Validates that the agent fixed the magnitude bias.
    """
    with open(OUTPUT_PATH, 'r') as f:
        results = json.load(f)
    
    recall = results.get("recall", 0.0)
    
    # Load data to double check they didn't just hardcode the GT
    data = np.load(DATA_PATH)
    gt = data['ground_truth']
    preds = np.array(results['predictions'])
    
    # Re-calculate recall externally to prevent spoofing the JSON 'recall' field
    # The submission must contain the actual list of IDs
    calculated_recall = np.mean(preds == gt)
    
    print(f"Verified Recall: {calculated_recall:.2%}")
    
    # The threshold is high because with Cosine Sim, the problem is trivial (1.0 recall).
    # With Dot Product, it is ~0.0.
    assert calculated_recall > 0.95, \
        f"FAILED: Recall is too low ({calculated_recall:.2%}). " \
        "The search is still biased by vector magnitude. " \
        "Ensure you L2-normalize the document vectors before dot product."

def test_code_logic_check():
    """
    Implicit check:
    We verify the predictions are not just the 'High Magnitude' docs.
    """
    data = np.load(DATA_PATH)
    docs = data['docs']
    
    # Calculate magnitudes of predicted docs
    with open(OUTPUT_PATH, 'r') as f:
        results = json.load(f)
    preds = np.array(results['predictions'])
    
    predicted_vectors = docs[preds]
    magnitudes = np.linalg.norm(predicted_vectors, axis=1)
    
    # The correct answers have magnitude ~1.0.
    # The trap answers have magnitude > 20.0.
    avg_mag = np.mean(magnitudes)
    
    assert avg_mag < 5.0, \
        f"FAILED: The documents you returned have huge average magnitude ({avg_mag:.2f}). " \
        "This confirms you are still sorting by raw Dot Product score without normalization."