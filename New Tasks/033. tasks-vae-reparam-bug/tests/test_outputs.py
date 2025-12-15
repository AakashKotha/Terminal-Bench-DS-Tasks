import pytest
import torch
import sys
import os

# Add task deps to path
sys.path.append("/app/task-deps")
try:
    from data import generate_synthetic_data
    # Attempt to import solution first, fallback to original if not present (will fail logic checks)
    sys.path.append("/app")
    if os.path.exists("/app/solution_vae.py"):
        from solution_vae import VAE, loss_function
        print("Imported VAE from solution_vae.py")
    else:
        # If user overwrote the original file directly
        from vae import VAE, loss_function
        print("Imported VAE from vae.py")
except ImportError as e:
    pytest.fail(f"Could not import model. Ensure /app/solution_vae.py or /app/task-deps/vae.py exists. Error: {e}")

def test_vae_convergence():
    """
    Trains the VAE for 100 steps. 
    If reparameterization is fixed, loss should drop significantly.
    If buggy, loss stays high or explodes.
    """
    torch.manual_seed(42)
    input_dim = 20
    model = VAE(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    data = generate_synthetic_data(n_samples=200, input_dim=input_dim)
    
    model.train()
    
    initial_loss = float('inf')
    final_loss = 0
    
    # Short training loop
    for epoch in range(100):
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()
        
        if epoch == 0:
            initial_loss = loss.item()
        final_loss = loss.item()

    print(f"Initial Loss: {initial_loss}")
    print(f"Final Loss: {final_loss}")
    
    # Check for improvement
    # With the bug, loss usually starts high and stays high or becomes NaN
    if torch.isnan(torch.tensor(final_loss)):
        pytest.fail("Training resulted in NaN loss. The math is likely invalid (e.g. log of negative).")
        
    # Valid VAE on this simple data should drop loss by at least 20-30% quickly
    improvement = initial_loss - final_loss
    if improvement < 100: 
        pytest.fail(f"Model did not learn. Loss improved by only {improvement}. Did you fix the std = exp(0.5*logvar) calculation?")

    print("Test Passed: Model is learning successfully.")

def test_code_logic():
    """
    Directly checks the reparameterize logic by inspecting the file or output behavior.
    """
    model = VAE()
    mu = torch.zeros(1, 2)
    # logvar = 0 -> std = 1. 
    logvar = torch.zeros(1, 2) 
    
    # If logic is std = logvar, then std = 0.
    # z = mu + std*eps = 0 + 0 = 0.
    # Output will be identically zero (deterministic).
    
    # If logic is std = exp(0.5*logvar) -> exp(0) = 1.
    # z = 0 + 1*eps = random.
    
    torch.manual_seed(999)
    z1 = model.reparameterize(mu, logvar)
    z2 = model.reparameterize(mu, logvar)
    
    if torch.equal(z1, z2):
        # If z1 == z2, it means no randomness was added (std was likely 0).
        pytest.fail("Reparameterize is deterministic! It seems std is being calculated as 0 (likely std=logvar where logvar=0). It should be stochastic.")