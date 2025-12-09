import numpy as np
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

OUTPUT_DIR = "/app/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_fl_data():
    print("Generating Federated Data partitions...")
    np.random.seed(42)
    
    # 1. Generate Dataset
    # 1000 samples, 20 features, 2 classes
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                               n_redundant=5, random_state=42)
    
    # Split Train/Test (Global Test Set)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Partition for 10 Clients (IID partition)
    n_clients = 10
    client_data = {}
    
    # Simple sharding
    shard_size = len(X_train) // n_clients
    for i in range(n_clients):
        start = i * shard_size
        end = (i + 1) * shard_size
        client_data[f"client_{i}"] = {
            "X": X_train[start:end],
            "y": y_train[start:end]
        }
        
    # 3. Save
    np.savez_compressed(os.path.join(OUTPUT_DIR, "client_partitions.npz"), **client_data)
    np.savez_compressed(os.path.join(OUTPUT_DIR, "global_test.npz"), X=X_test, y=y_test)
    
    print(f"Data generated. {n_clients} clients, {len(X_test)} test samples.")

if __name__ == "__main__":
    generate_fl_data()