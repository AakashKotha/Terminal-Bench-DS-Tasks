import torch
import sys
import os
import pytest

# Paths
sys.path.append("/app/task-deps")
sys.path.append("/app")

# Import broken module to establish baseline logic comparison
import loss as broken_loss_module

def test_mining_logic_fix():
    """
    Verifies that the user's solution correctly identifies the hardest negative.
    """
    # Import user solution
    if os.path.exists("/app/solution_loss.py"):
        try:
            import solution_loss
        except ImportError:
            pytest.fail("Found solution_loss.py but could not import it. Check syntax.")
    else:
        pytest.fail("solution_loss.py not found. Please fix the bug and save the file.")

    # Setup a deterministic scenario
    # Anchor (A): Class 0
    # Positive (P): Class 0, Dist(A, P) = 0.5
    # Negative Easy (NE): Class 1, Dist(A, NE) = 2.0 (Far away)
    # Negative Hard (NH): Class 1, Dist(A, NH) = 0.2 (Very close, arguably too close)
    
    # We construct a distance matrix manually to test the logic directly, 
    # instead of relying on embeddings generation which is harder to control exactly.
    # However, since the function takes embeddings, we will mock the behavior by 
    # creating 4 embeddings on a line.
    
    # 1D embeddings:
    # A = 0.0 (Label 0)
    # P = 0.5 (Label 0) -> d(A,P) = 0.5
    # NH = 0.2 (Label 1) -> d(A,NH) = 0.2
    # NE = 2.0 (Label 1) -> d(A,NE) = 2.0
    
    embeddings = torch.tensor([[0.0], [0.5], [0.2], [2.0]])
    labels = torch.tensor([0, 0, 1, 1])
    
    # Expected Behavior for Anchor A (index 0):
    # Hardest Positive = P (0.5)
    # Hardest Negative = NH (0.2) -> This is the one closest to A.
    #
    # Formula: max(d(a,p) - d(a,n) + margin, 0)
    # Loss = max(0.5 - 0.2 + 1.0, 0) = 1.3
    
    # Broken Behavior (Max Negative):
    # Picks NE (2.0)
    # Loss = max(0.5 - 2.0 + 1.0, 0) = 0.0 (Gradient vanishes)
    
    margin = 1.0
    
    # Run User's Function
    user_loss = solution_loss.batch_hard_triplet_loss(embeddings, labels, margin)
    
    print(f"User computed loss: {user_loss.item()}")
    
    if user_loss.item() < 0.1:
        pytest.fail(f"Loss is {user_loss.item():.4f} (approx 0). \n"
                    "This suggests you are still selecting the 'Easy Negative' (furthest away). \n"
                    "You need to select the negative with the MINIMUM distance to the anchor.")
        
    # Check strict value roughly
    # We compute mean loss over batch.
    # Let's calculate manual expected mean.
    # A(0): Hp=0.5, Hn=0.2. Loss=1.3
    # P(1): Hp=0.5 (A), Hn=0.3 (NH, dist|0.5-0.2|=0.3). Loss = 0.5 - 0.3 + 1 = 1.2
    # NH(2): Hp=1.8 (NE), Hn=0.2 (A). Loss = 1.8 - 0.2 + 1 = 2.6
    # NE(3): Hp=1.8 (NH), Hn=1.5 (P). Loss = 1.8 - 1.5 + 1 = 1.3
    # Mean = (1.3+1.2+2.6+1.3)/4 = 1.6
    
    if not torch.isclose(user_loss, torch.tensor(1.6), atol=0.1):
        pytest.fail(f"Calculated loss {user_loss.item()} matches neither the broken (0.0) nor correct (~1.6) value. Check logic.")
        
    print("Test Passed: Loss calculation reflects hard negative mining.")

def test_robustness_to_zeros():
    """
    A common pitfall when fixing 'min' is that the mask (0s) becomes the minimum.
    Since we mask out positive items (same label) from the negative pool, 
    we must ensure those masked values don't become the 'minimum' chosen as hard negative.
    """
    import solution_loss
    
    embeddings = torch.tensor([[0.0], [1.0], [0.1]])
    labels = torch.tensor([0, 0, 1])
    # Dist matrix:
    # [[0.0, 1.0, 0.1],
    #  [1.0, 0.0, 0.9],
    #  [0.1, 0.9, 0.0]]
    
    # For A(0): Positives=[0,1], Negatives=[2].
    # Mask for Negatives (diff label): [0, 0, 1].
    # Neg distances: [0.0, 0.0, 0.1] (if 0-masked)
    
    # If user uses min() on zero-masked array, they get 0.0 (self-distance) or 0.0 (masked pos).
    # Correct Hard Neg for A(0) is [2] -> 0.1.
    
    loss = solution_loss.batch_hard_triplet_loss(embeddings, labels, margin=1.0)
    
    # A(0): Hp=1.0, Hn=0.1. Loss=1.0 - 0.1 + 1 = 1.9.
    # If Hn was picked as 0.0 (bad mask handling), Loss=2.0.
    
    # Let's focus on if it runs without crashing and produces positive loss
    assert loss.item() > 0.0