import torch
from dataset import CustomDataset
import os
import sys

# Simple IoU Calculator for debugging
def box_iou_batch(boxes_a, boxes_b):
    """
    Computes IoU between two sets of boxes.
    Assumes boxes are in [cx, cy, w, h] normalized format? 
    No, standard IoU usually works on [x1, y1, x2, y2].
    So we must convert BACK to corners to check overlap.
    """
    def xywh2xyxy(x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
        y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
        y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
        y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
        return y

    # We assume inputs are Normalized CXCYWH (Target format)
    # So we un-normalize and convert to corners for IoU calculation
    # Just for metric calculation logic
    
    # Actually, simpler check:
    # If the dataset is correct, its output matches the ground truth transformed.
    # We don't need full IoU implementation if we have exact math check.
    pass

def run_check():
    print("Initializing Dataset...")
    ds = CustomDataset("/app/data/labels.csv", "/app/data/images")
    
    # We load row 0 manually to calculate ground truth
    import pandas as pd
    df = pd.read_csv("/app/data/labels.csv")
    row0 = df.iloc[0]
    
    # GT Absolute
    x1, y1, x2, y2 = row0['x1'], row0['y1'], row0['x2'], row0['y2']
    W, H = 1280, 720
    
    # Calculate Expected Target (YOLO Normalized)
    # cx = ((x1 + x2) / 2) / W
    # cy = ((y1 + y2) / 2) / H
    # w  = (x2 - x1) / W
    # h  = (y2 - y1) / H
    
    expected_cx = ((x1 + x2) / 2) / W
    expected_cy = ((y1 + y2) / 2) / H
    expected_w = (x2 - x1) / W
    expected_h = (y2 - y1) / H
    
    expected = torch.tensor([expected_cx, expected_cy, expected_w, expected_h])
    
    print(f"Row 0 Raw: {x1}, {y1}, {x2}, {y2}")
    print(f"Expected YOLO Normalized: {expected}")
    
    # Get from Dataset
    _, output_box, _ = ds[0]
    print(f"Dataset Output: {output_box}")
    
    # Check bounds (Basic Sanity)
    if output_box.max() > 1.0:
        print("FAILURE: Dataset output contains values > 1.0. You forgot to normalize!")
        return
        
    # Check diff
    diff = (output_box - expected).abs().sum()
    if diff > 0.001:
        print(f"FAILURE: Output mismatch. Diff: {diff:.4f}")
        print("Did you convert (x1,x2) -> center_x and (x2-x1) -> width correctly?")
    else:
        print("SUCCESS: Transforms look correct.")
        
        # Generate artifact for the rigorous test runner
        # Save a batch of 10
        batch_boxes = []
        for i in range(10):
            _, b, _ = ds[i]
            batch_boxes.append(b)
        
        torch.save(torch.stack(batch_boxes), "/app/output/validated_batch.pt")
        print("Saved /app/output/validated_batch.pt")

if __name__ == "__main__":
    run_check()