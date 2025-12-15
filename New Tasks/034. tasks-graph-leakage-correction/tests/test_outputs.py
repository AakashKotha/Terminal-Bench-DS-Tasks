import torch
import sys
import os
import pytest

# Add paths
sys.path.append("/app/task-deps")
sys.path.append("/app")

# Try importing solution, else fail
try:
    if os.path.exists("/app/solution_gnn.py"):
        from solution_gnn import GATLayer
        print("Testing solution_gnn.py...")
    else:
        from leaky_gnn import GATLayer
        print("Testing original leaky_gnn.py...")
except ImportError:
    pytest.fail("Could not import GATLayer. Ensure solution_gnn.py exists.")

from generate_data import get_disconnected_graph

def test_information_leakage():
    """
    We verify GNN correctness by checking gradients.
    If Node 2 is not connected to Node 0, then changing Node 2's input features
    should have ZERO effect on Node 0's output features.
    
    Mathematically: d(Output_0) / d(Input_2) == 0
    """
    feats, adj = get_disconnected_graph()
    
    model = GATLayer(in_features=10, out_features=5)
    
    # Forward pass
    out = model(feats, adj)
    
    # We want to check if Node 2 influenced Node 0.
    # We take the sum of Node 0's output vector as the scalar loss.
    target_scalar = out[0].sum()
    
    # Compute gradients of this scalar w.r.t input features
    grad_inputs = torch.autograd.grad(target_scalar, feats, create_graph=False)[0]
    
    # grad_inputs shape is (3, 10) corresponding to nodes 0, 1, 2.
    # grad_inputs[2] represents d(Target) / d(Feats_2).
    # Since Target depends only on Node 0 output, and Node 0 is NOT connected to Node 2,
    # this gradient must be 0.
    
    gradient_leakage = grad_inputs[2].abs().sum().item()
    
    print(f"Gradient leakage from Node 2 to Node 0: {gradient_leakage}")
    
    if gradient_leakage > 1e-4:
        pytest.fail(f"TEST FAILED: Information Leakage Detected! \n"
                    f"Node 0 is reacting to Node 2's inputs (Grad: {gradient_leakage}).\n"
                    f"This means the graph structure (adj_matrix) is being ignored or masked incorrectly.")
        
    print("TEST PASSED: No information leakage. The layer respects graph topology.")

def test_adjacency_mask_usage():
    """
    Anti-cheat: ensure they didn't just hardcode zero gradients or something strange.
    We check if connected nodes DO have gradients.
    """
    feats, adj = get_disconnected_graph()
    model = GATLayer(in_features=10, out_features=5)
    out = model(feats, adj)
    
    # Check 0 -> 1 connection (Node 0 output should depend on Node 1 input)
    target_scalar = out[0].sum()
    grad_inputs = torch.autograd.grad(target_scalar, feats)[0]
    
    gradient_flow = grad_inputs[1].abs().sum().item()
    
    if gradient_flow < 1e-6:
        pytest.fail("TEST FAILED: No gradient flow between connected neighbors (0 and 1). Did you zero out everything?")
        
    print(f"Gradient flow validation: {gradient_flow} (OK)")