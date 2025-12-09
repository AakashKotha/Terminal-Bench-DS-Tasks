import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

# Define a simple GCN Layer from scratch (First Principles)
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj):
        # Message Passing: A * X * W
        # 1. Aggregate neighbors (A * X)
        support = torch.mm(adj, x)
        # 2. Transform (Linear W)
        output = self.linear(support)
        return output

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GraphConvLayer(in_dim, hidden_dim)
        self.conv2 = GraphConvLayer(hidden_dim, out_dim)
        
    def forward(self, x, adj):
        x = F.relu(self.conv1(x, adj))
        x = self.conv2(x, adj)
        return F.log_softmax(x, dim=1)

def load_data():
    with open("/app/data/graph.json", "r") as f:
        data = json.load(f)
        
    x = torch.tensor(data["x"])
    y = torch.tensor(data["y"])
    train_mask = torch.tensor(data["train_mask"])
    test_mask = torch.tensor(data["test_mask"])
    
    # --- LEAKAGE ZONE START ---
    
    # The script loads ALL edges into the graph structure
    all_edges = torch.tensor(data["edge_index"])
    
    num_nodes = x.shape[0]
    adj = torch.zeros((num_nodes, num_nodes))
    
    # Populate adjacency with ALL edges (Train AND Test)
    # This allows test nodes to exchange info during training loops
    adj[all_edges[0], all_edges[1]] = 1.0
    
    # Add self-loops and normalize (Standard GCN preprocessing)
    adj = adj + torch.eye(num_nodes)
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm_adj = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
    
    # --- LEAKAGE ZONE END ---
    
    return x, y, norm_adj, train_mask, test_mask, all_edges

def train():
    x, y, adj, train_mask, test_mask, _ = load_data()
    
    model = GCN(in_dim=16, hidden_dim=32, out_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("Training GNN...")
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        output = model(x, adj)
        loss = F.nll_loss(output[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
        
    # Evaluation
    model.eval()
    output = model(x, adj)
    pred = output.argmax(dim=1)
    
    acc = (pred[test_mask] == y[test_mask]).float().mean()
    print(f"Final Test Accuracy: {acc.item():.4f}")
    
    # Save artifacts (as requested by task)
    os.makedirs("/app/output", exist_ok=True)
    
    # Save Metrics
    with open("/app/output/metrics.json", "w") as f:
        json.dump({"test_acc": float(acc)}, f)
        
    # Save Adjacency (The validator will check this for leakage)
    torch.save(adj, "/app/output/clean_adj.pt")

if __name__ == "__main__":
    train()