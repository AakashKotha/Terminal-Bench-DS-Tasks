import csv
import os
import random
import torch

OUTPUT_DIR = "/app/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_dataset():
    print("Generating Synthetic Bounding Box Data...")
    
    # Image Specs (Non-square to catch width/height swap errors)
    IMG_W = 1280
    IMG_H = 720
    N_SAMPLES = 100
    
    csv_path = os.path.join(OUTPUT_DIR, "labels.csv")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "x1", "y1", "x2", "y2", "class_id"])
        
        for i in range(N_SAMPLES):
            img_name = f"img_{i:04d}.jpg"
            
            # Generate random box
            w = random.randint(50, 300)
            h = random.randint(50, 300)
            x1 = random.randint(0, IMG_W - w)
            y1 = random.randint(0, IMG_H - h)
            x2 = x1 + w
            y2 = y1 + h
            
            # Class ID (0-4)
            cid = random.randint(0, 4)
            
            # Write PASCAL VOC format (Absolute)
            writer.writerow([img_name, x1, y1, x2, y2, cid])
            
    print(f"Generated {N_SAMPLES} annotations in {csv_path}")
    print(f"Image Dimensions assumed: {IMG_W}x{IMG_H}")

if __name__ == "__main__":
    generate_dataset()