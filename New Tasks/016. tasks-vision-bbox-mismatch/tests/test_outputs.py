import torch
import pandas as pd
import pytest
import os

OUTPUT_PATH = "/app/output/validated_batch.pt"
CSV_PATH = "/app/data/labels.csv"

def test_artifact_generated():
    assert os.path.exists(OUTPUT_PATH), "Artifact /app/output/validated_batch.pt not found. Did you run the debug script successfully?"

def test_normalization_bounds():
    """
    Are all coordinates between 0 and 1?
    """
    tensor = torch.load(OUTPUT_PATH)
    assert tensor.min() >= 0.0, "Found negative coordinates."
    assert tensor.max() <= 1.0, "Found coordinates > 1.0. Normalization failed."

def test_geometric_correctness():
    """
    Re-calculates the transform from source CSV and compares with agent output.
    """
    df = pd.read_csv(CSV_PATH)
    agent_tensor = torch.load(OUTPUT_PATH) # Shape [10, 4]
    
    IMG_W = 1280
    IMG_H = 720
    
    for i in range(10):
        row = df.iloc[i]
        # Source (Pascal VOC Absolute)
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        
        # Target (YOLO Normalized)
        # cx
        gt_cx = ((x1 + x2) / 2.0) / IMG_W
        # cy
        gt_cy = ((y1 + y2) / 2.0) / IMG_H
        # w
        gt_w = (x2 - x1) / IMG_W
        # h
        gt_h = (y2 - y1) / IMG_H
        
        agent_box = agent_tensor[i]
        
        # We verify each component individually to give helpful errors
        assert torch.isclose(agent_box[0], torch.tensor(gt_cx), atol=1e-4), \
            f"Index {i}: Center X mismatch. Did you divide by width {IMG_W}?"
            
        assert torch.isclose(agent_box[1], torch.tensor(gt_cy), atol=1e-4), \
            f"Index {i}: Center Y mismatch. Did you divide by height {IMG_H}?"
            
        assert torch.isclose(agent_box[2], torch.tensor(gt_w), atol=1e-4), \
            f"Index {i}: Width mismatch. formula should be (x2-x1)/W."
            
        assert torch.isclose(agent_box[3], torch.tensor(gt_h), atol=1e-4), \
            f"Index {i}: Height mismatch."