import json
import os
import pytest

OUTPUT_FILE = "best_config.json"
LOG_FILE = "usage_log.txt"

def test_output_file_exists():
    assert os.path.exists(OUTPUT_FILE), f"The file {OUTPUT_FILE} was not found."

def test_budget_constraint():
    """Ensure the agent didn't simply brute force thousands of combinations."""
    if not os.path.exists(LOG_FILE):
        # If no log file, it implies mock_trainer wasn't run at all or file deleted
        pytest.fail("usage_log.txt not found. Did the solution run task_deps/mock_trainer.py?")
    
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
    
    call_count = len(lines)
    # Budget is 50, but we verify 60 to be lenient on restarts or tiny bugs
    assert call_count <= 60, f"Solution exceeded budget! Used {call_count} calls, limit is 50."
    assert call_count > 0, "Solution did not call the training script at all."

def test_result_quality():
    """Verify the found configuration actually yields high accuracy."""
    with open(OUTPUT_FILE, "r") as f:
        data = json.load(f)
    
    assert "accuracy" in data, "JSON output missing 'accuracy' field"
    assert "optimizer" in data, "JSON output missing 'optimizer' field"
    assert "learning_rate" in data, "JSON output missing 'learning_rate' field"
    
    # We verify the reported accuracy is actually high
    accuracy = data["accuracy"]
    assert accuracy > 0.95, f"Best accuracy found ({accuracy}) is below the target of 0.95"
    
    # We also verify the agent isn't hallucinating the score by recalculating it
    # We essentially re-implement a simplified check of the blackbox here strictly for verification
    # logic matching mock_trainer.py for 'adam'
    if data["optimizer"] == "adam":
        # Rough check logic from mock_trainer
        import math
        lr_penalty = min(abs(math.log10(data["learning_rate"]) - math.log10(0.001)), 3.0) * 0.2
        layer_penalty = abs(data["layers"] - 3) * 0.05
        expected_base = 0.99 - lr_penalty - layer_penalty
        # Allow small margin for random noise in the trainer
        assert accuracy >= (expected_base - 0.03), "Reported accuracy seems inconsistent with parameters provided."
    elif data["optimizer"] == "sgd":
        assert accuracy <= 0.87, "SGD should not achieve > 0.95 accuracy in this simulation."