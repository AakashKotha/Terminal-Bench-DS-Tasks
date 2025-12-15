import pytest
import numpy as np
import sys
import os
import subprocess

# Add app to path to import nanograd
sys.path.append("/app")

try:
    from nanograd import Tensor
except ImportError:
    pass # Will fail in environments where nanograd isn't present, but that's handled by the runner

def test_training_script_converges():
    """
    Runs the train.py script and asserts it exits with 0 and prints success.
    Also checks if loss is reasonably low.
    """
    result = subprocess.run(
        ["python3", "/app/train.py"], 
        capture_output=True, 
        text=True
    )
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    assert result.returncode == 0, "Training script failed (crashed)."
    assert "CRITICAL ERROR" not in result.stdout, "Gradient shape mismatch detected."
    
    # Check if final loss is reasonable (it starts around 200-500 usually)
    lines = result.stdout.strip().split('\n')
    last_loss_line = [l for l in lines if "Step 49" in l]
    assert last_loss_line, "Could not find final training step output"
    
    loss_val = float(last_loss_line[0].split("loss ")[1])
    assert loss_val < 10.0, f"Model failed to converge, final loss: {loss_val} (Expected < 10.0)"

def test_autograd_add_broadcasting():
    """
    Unit test specifically for the broadcasting bug.
    y = x + b
    x shape (2, 3)
    b shape (3,)
    dy/db should be shape (3,) and contain sum over axis 0.
    """
    x_np = np.random.randn(2, 3).astype(np.float32)
    b_np = np.random.randn(3).astype(np.float32)
    
    x = Tensor(x_np)
    b = Tensor(b_np)
    
    # Forward
    y = x + b
    
    # Backward
    # Let's say dL/dy is all ones
    y.grad = np.ones_like(y.data)
    y._backward()
    
    # Check b.grad
    # Correct math: b.grad should be sum(y.grad, axis=0) -> shape (3,) with values [2., 2., 2.]
    expected_grad = np.sum(y.grad, axis=0)
    
    assert b.grad.shape == b_np.shape, f"Gradient shape mismatch. Expected {b_np.shape}, got {b.grad.shape}"
    np.testing.assert_allclose(b.grad, expected_grad, rtol=1e-5, err_msg="Gradient values are incorrect (likely failed to sum over broadcasted dimension)")

def test_autograd_mul_broadcasting():
    """
    Unit test for multiplication broadcasting.
    y = x * b
    x shape (2, 3)
    b shape (3,)
    """
    x_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    b_np = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    
    x = Tensor(x_np)
    b = Tensor(b_np)
    
    y = x * b # [[0.5, 1, 1.5], [2, 2.5, 3]]
    
    y.grad = np.ones_like(y.data)
    y._backward()
    
    # d(x*b)/db = x
    # grad_b = sum(x * grad_y, axis=0)
    expected_grad = np.sum(x_np * y.grad, axis=0) # [1+4, 2+5, 3+6] = [5, 7, 9]
    
    assert b.grad.shape == b_np.shape, f"Mul Gradient shape mismatch. Expected {b_np.shape}, got {b.grad.shape}"
    np.testing.assert_allclose(b.grad, expected_grad, rtol=1e-5)