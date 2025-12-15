import torch
import numpy as np
import os
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

DATA_DIR = "/app/data"
os.makedirs(DATA_DIR, exist_ok=True)

def generate_dataset():
    print("Generating Synthetic Dataset...")
    # 20 features, 5 classes. 
    # Hard separation: High cluster_std means classes overlap significantly.
    X, y = make_blobs(n_samples=2000, n_features=20, centers=5, cluster_std=2.5, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to Tensor
    data = {
        "X_train": torch.FloatTensor(X_train),
        "y_train": torch.LongTensor(y_train),
        "X_test": torch.FloatTensor(X_test),
        "y_test": torch.LongTensor(y_test)
    }
    
    torch.save(data, os.path.join(DATA_DIR, "dataset.pt"))
    print("Dataset saved to dataset.pt")

if __name__ == "__main__":
    generate_dataset()