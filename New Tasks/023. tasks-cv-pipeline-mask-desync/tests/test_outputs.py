import pytest
import os
import json
import torch
import sys
from PIL import Image

sys.path.append("/app/src")
from dataset import SegmentationDataset

STATUS_FILE = "/app/output/status.json"

def test_status_file_exists():
    assert os.path.exists(STATUS_FILE), "Validation script did not produce output. Did you run /app/src/validate_alignment.py?"

def test_iou_score():
    with open(STATUS_FILE, "r") as f:
        data = json.load(f)
    assert data["iou"] > 0.99, f"IoU is too low ({data['iou']}). Mask/Image alignment failed."

def test_augmentations_still_active():
    """
    The agent might cheat by removing RandomCrop/Flip entirely.
    If they do that, the output is always the center crop or raw image.
    We check if there is variance in the output.
    """
    ds = SegmentationDataset(size=20)
    
    # Fetch the same index multiple times. 
    # If random transforms are active, we should get DIFFERENT results for the same index
    # (assuming the Dataset generates fresh random transforms on call, which standard PyTorch Transforms do)
    # However, standard Dataset implementations usually instantiate transforms in __init__. 
    # If the agent fixed it by functional transforms inside __getitem__, randomness should persist per call.
    
    tpl1, _ = ds[0]
    tpl2, _ = ds[0]
    tpl3, _ = ds[0]
    
    # Stack them
    stack = torch.stack([tpl1, tpl2, tpl3])
    
    # Calculate variance across the calls
    variance = torch.var(stack.float())
    
    # If variance is 0, it means ds[0] always returns the exact same tensor.
    # This implies the agent replaced RandomCrop with CenterCrop or removed augmentations.
    # Note: There's a small chance random crop picks same spot 3 times, but very unlikely with 256->128 crop.
    
    assert variance > 0.0, \
        "FAILED: Data Augmentation appears to be disabled. You must keep RandomCrop/Flip active, just synchronized."

def test_synchronization_logic():
    """
    Manual verification of the fix logic.
    We expect the agent to use functional transforms (TF.crop) or get_params.
    """
    ds = SegmentationDataset(size=5)
    img, mask = ds[0]
    
    # Check strict equality (since synthetic data was identical)
    assert torch.allclose(img, mask, atol=1e-5), \
        "Image and Mask tensors differ. Synchronization failed."