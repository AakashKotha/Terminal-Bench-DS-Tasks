import torch

def get_disconnected_graph():
    """
    Creates a simple graph with 3 nodes.
    0 -- 1    (Nodes 0 and 1 are connected)
    2         (Node 2 is isolated)
    
    Returns:
        feats: (3, 10)
        adj: (3, 3)
    """
    torch.manual_seed(42)
    feats = torch.randn(3, 10, requires_grad=True)
    
    # Adjacency matrix (symmetric, including self-loops)
    adj = torch.eye(3)
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    # Node 2 is strictly 0 connections to 0 and 1
    
    return feats, adj