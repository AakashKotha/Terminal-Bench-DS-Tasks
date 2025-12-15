import torch
import sys
import json
import os
import numpy as np

# Import the dataset
sys.path.append("/app/src")
from dataset import SegmentationDataset

OUTPUT_FILE = "/app/output/status.json"

def calculate_iou(pred, target):
    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Threshold at 0.5 (images are 0 or 1 floats)
    pred_bin = (pred > 0.5).int()
    target_bin = (target > 0.5).int()
    
    intersection = (pred_bin & target_bin).sum().item()
    union = (pred_bin | target_bin).sum().item()
    
    if union == 0:
        return 1.0 # Both empty
    
    return intersection / union

def run_validation():
    print("Initializing Dataset...")
    ds = SegmentationDataset(size=50)
    
    ious = []
    
    print(f"Checking {len(ds)} samples for geometric alignment...")
    
    for i in range(len(ds)):
        img, mask = ds[i]
        
        # In our synthetic data, the image and mask start identical.
        # If the transforms are synchronized, the resulting tensors should be identical.
        # (Allowing for tiny float diffs, checking IoU is robust).
        
        score = calculate_iou(img, mask)
        ious.append(score)
        
        if i < 5:
            print(f"Sample {i}: IoU = {score:.4f}")

    avg_iou = np.mean(ious)
    print(f"\nAverage IoU: {avg_iou:.4f}")
    
    if avg_iou > 0.99:
        print("SUCCESS: Transforms are synchronized!")
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "w") as f:
            json.dump({"status": "success", "iou": avg_iou}, f)
    else:
        print("FAILURE: Transforms are desynchronized. The mask does not match the image.")
        print("Hint: transforms.RandomCrop generates a NEW random coordinate every time it is called.")

if __name__ == "__main__":
    run_validation()