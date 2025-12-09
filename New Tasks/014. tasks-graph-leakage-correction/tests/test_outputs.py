import torch
import json
import os
import pytest

ADJ_PATH = "/app/output/clean_adj.pt"
METRICS_PATH = "/app/output/metrics.json"
DATA_PATH = "/app/data/graph.json"

def test_artifacts_exist():
    assert os.path.exists(ADJ_PATH), "Adjacency matrix artifact not found."
    assert os.path.exists(METRICS_PATH), "Metrics file not found."

def test_no_leakage_in_adjacency():
    """
    CRITICAL CHECK:
    We load the 'clean_adj.pt' produced by the agent.
    We verify that it does NOT contain edges connecting Test nodes.
    """
    # 1. Load Ground Truth Data to identify 'Illegal' edges
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    
    # Convert masks to indices
    train_mask = torch.tensor(data["train_mask"])
    test_mask = torch.tensor(data["test_mask"])
    test_indices = torch.where(test_mask)[0]
    
    edge_index = torch.tensor(data["edge_index"])
    
    # Identify illegal edges: An edge is illegal if BOTH source and target are in Test Set
    # (Strict Inductive) OR if the edge connects a Test Node to a Train Node (Transductive masking).
    # For this task, the most robust check for "Fixing Leakage" is ensuring
    # that the adjacency matrix assumes Test nodes are isolated or only connected via 
    # explicitly allowed paths. 
    
    # Simple rigorous check: The agent should have filtered 'all_edges' based on 'train_mask'.
    # So if we look at the saved Adjacency Matrix:
    # A[u, v] should be 0 if 'u' is a Test Node (assuming standard GCN split where test nodes are hidden during training steps).
    # Wait, usually in GCN (semi-supervised), we keep nodes but hide labels.
    # BUT, to prevent leakage in this specific "Hard" task, we define leakage as "Using Test Edges".
    
    # Let's check specific known test edges.
    # Get all edges that connect two test nodes
    src = edge_index[0]
    dst = edge_index[1]
    
    # Mask of edges where BOTH ends are test nodes
    is_test_edge = test_mask[src] & test_mask[dst]
    test_edge_indices = torch.where(is_test_edge)[0]
    
    # Load Agent's Matrix
    adj_matrix = torch.load(ADJ_PATH)
    
    # Check if these edges exist in the matrix (value > 0)
    # We sample a few illegal edges
    leaked_count = 0
    check_count = 0
    
    for idx in test_edge_indices[:100]: # Check first 100 illegal edges
        u, v = src[idx].item(), dst[idx].item()
        if adj_matrix[u, v] > 0:
            leaked_count += 1
        check_count += 1
            
    assert leaked_count == 0, \
        f"FAILED: Data Leakage Detected. Found {leaked_count}/{check_count} illegal edges in Adjacency Matrix.\n" \
        "Edges connecting two Test nodes must be removed from the graph structure before convolution."

def test_accuracy_reality_check():
    """
    If the agent fixed the leakage, accuracy should drop from 1.0 to something reasonable.
    """
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    
    acc = metrics["test_acc"]
    
    print(f"Reported Accuracy: {acc}")
    
    assert acc < 0.95, \
        f"FAILED: Accuracy is still too high ({acc}). This implies leakage persists."
        
    assert acc > 0.55, \
        f"FAILED: Accuracy dropped too low ({acc}). The model stopped learning entirely."