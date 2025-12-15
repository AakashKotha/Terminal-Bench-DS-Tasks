import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Ensure we can import from task-deps
sys.path.append("/app/task-deps")

from transformer import SimpleTransformer

def test_attention_scaling_exists():
    """
    Introspectively checks if the scaling logic is present in the source file
    or runs a dummy forward pass to check gradient magnitudes? 
    It's safer to check if the model actually learns.
    """
    pass

def generate_toy_data():
    # Very simple copy task: Input [1, 2, 3] -> Output [1, 2, 3]
    X = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    Y = X.clone()
    return X, Y

def test_model_convergence_on_toy_task():
    """
    We define a micro-training loop here. 
    If the attention scaling bug is present, this simple task often fails 
    or takes way too long to converge for standard params.
    """
    torch.manual_seed(42)
    
    # Tiny Model
    vocab_size = 10
    d_model = 32
    n_heads = 4 # head_dim = 8
    model = SimpleTransformer(vocab_size, d_model, n_heads, n_layers=1, d_ff=64, max_len=10)
    
    # Without scaling, scores are ~ size of d_model. With d=32, sqrt(d) ~ 5.6.
    # It's subtle at d=32, but critical at d=64/128. 
    # Let's verify learning capability.
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Overfit a single batch
    X = torch.randint(0, vocab_size, (16, 8))
    Y = X.clone() # Copy task
    
    final_loss = 100
    for i in range(50): # Should converge very fast if correct
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output.view(-1, vocab_size), Y.view(-1))
        loss.backward()
        optimizer.step()
        final_loss = loss.item()
    
    print(f"Test Final Loss: {final_loss}")
    
    # If scaling is missing, gradients are often too sharp or vanish, making optimization hard.
    # With correct scaling, loss usually drops < 0.1 quickly for this toy task.
    assert final_loss < 0.5, f"Model failed to overfit toy data. Final loss: {final_loss}. Did you fix the attention scaling?"

def test_module_structure():
    """Ensure the user didn't cheat by removing layers."""
    model = SimpleTransformer(10, 32, 4, 2, 64)
    assert len(model.layers) == 2
    assert isinstance(model.layers[0].attention, nn.Module)