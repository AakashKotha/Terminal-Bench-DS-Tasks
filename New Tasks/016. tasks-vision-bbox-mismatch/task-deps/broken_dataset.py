import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        # Fixed dimensions for this dataset
        self.img_w = 1280
        self.img_h = 720

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # --- BROKEN LOGIC START ---
        
        # The CSV contains: x1, y1, x2, y2 (Absolute Pixels)
        # The Model expects: x_center, y_center, width, height (Normalized 0-1)
        
        # CURRENT (WRONG) BEHAVIOR:
        # Returning raw absolute coordinates directly.
        # This will fail because values are > 1.0, and represent corners, not center.
        
        boxes = torch.tensor([
            row['x1'], 
            row['y1'], 
            row['x2'], 
            row['y2']
        ], dtype=torch.float32)
        
        # --- BROKEN LOGIC END ---
        
        labels = torch.tensor(row['class_id'], dtype=torch.long)
        
        # Fake image tensor (not needed for this geometry task)
        img = torch.zeros((3, self.img_h, self.img_w))
        
        return img, boxes, labels