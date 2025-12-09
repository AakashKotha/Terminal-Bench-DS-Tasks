import torch
import os
import json
import numpy as np

OUTPUT_DIR = "/app/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_graph():
    print("Generating Crypto Graph Data...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    num_nodes = 500
    num_features = 16
    num_classes = 2 # Fraud vs Legit
    
    # 1. Generate Node Features (Random noise, slightly correlated with class)
    # The leakage will be the primary signal, features are weak.
    labels = torch.randint(0, num_classes, (num_nodes,))
    features = torch.randn(num_nodes, num_features)
    
    # Add small signal to features so model learns *something* without leakage
    for i in range(num_nodes):
        features[i] += labels[i] * 0.5
        
    # 2. Generate Edges (Homophily: Fraud connects to Fraud)
    # If we include these edges in Test, the model just checks the neighbor's label.
    edges = []
    
    for i in range(num_nodes):
        # Create 5 connections per node
        for _ in range(5):
            target = np.random.randint(0, num_nodes)
            # 80% chance to connect to same class (Leakage Source)
            if labels[i] == labels[target]:
                prob = 0.8
            else:
                prob = 0.2
                
            if np.random.rand() < prob and i != target:
                edges.append([i, target])
                edges.append([target, i]) # Undirected
                
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # 3. Split Data
    # 60% Train, 40% Test
    perm = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_idx = perm[:int(0.6 * num_nodes)]
    test_idx = perm[int(0.6 * num_nodes):]
    
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    
    # 4. Save
    # We save edges as a raw list. The agent must decide which ones to load.
    data_dict = {
        "x": features.tolist(),
        "y": labels.tolist(),
        "edge_index": edge_index.tolist(),
        "train_mask": train_mask.tolist(),
        "test_mask": test_mask.tolist()
    }
    
    with open(os.path.join(OUTPUT_DIR, "graph.json"), "w") as f:
        json.dump(data_dict, f)
        
    print(f"Graph generated: {num_nodes} nodes, {len(edges)} edges.")

if __name__ == "__main__":
    generate_graph()