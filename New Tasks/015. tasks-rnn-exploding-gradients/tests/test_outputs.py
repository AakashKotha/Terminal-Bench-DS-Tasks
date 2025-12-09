import torch
import os
import pytest
import sys

# Ensure we can verify the logic (optional introspection)
sys.path.append("/app/src")

MODEL_PATH = "/app/models/stable_rnn.pt"
TRAIN_SCRIPT = "/app/src/train.py"

def test_model_artifact_exists():
    assert os.path.exists(MODEL_PATH), "Model file not found. Did the training script complete?"

def test_weights_are_finite():
    """
    Check if the saved model contains NaNs or Infs.
    If the agent didn't fix the explosion, the saved weights will be garbage.
    """
    state_dict = torch.load(MODEL_PATH)
    
    for name, param in state_dict.items():
        assert torch.isfinite(param).all(), \
            f"Layer '{name}' contains NaN or Infinity values. Gradient Explosion was not fixed."

def test_loss_convergence():
    """
    We verify the training by running a simplified forward pass check.
    We re-run a dummy training step with the agent's code logic (via string inspection)
    OR we rely on the fact that if weights are finite after 5 epochs of High LR RNN,
    clipping MUST have been used.
    """
    # Inspect the source code for the fix
    with open(TRAIN_SCRIPT, 'r') as f:
        code = f.read()
    
    # We look for "clip_grad_norm" or "clip_grad_value"
    # This is a heuristic, but reliable for this specific problem type.
    has_clip = "clip_grad_norm" in code or "clip_grad_value" in code
    
    if not has_clip:
        # Fallback: Maybe they did manual clipping? 
        # (p.grad.data.clamp_)
        has_clamp = "clamp" in code
        assert has_clamp, \
            "FAILED: No Gradient Clipping detected. You must use 'torch.nn.utils.clip_grad_norm_' or clamp gradients manually to prevent explosion."

def test_simulated_training_stability():
    """
    Re-load the model and run a forward pass on new data to ensure it predicts valid numbers.
    """
    # Redefine model structure since we can't easily import the class if they modified structure (forbidden, but possible)
    # We assume they kept structure.
    try:
        from train import DeepRNN
    except ImportError:
        pytest.fail("Could not import DeepRNN from train.py")

    model = DeepRNN(5, 64, 5)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    dummy_input = torch.randn(10, 40, 5) # Batch 10, Seq 40, Dim 5
    with torch.no_grad():
        output = model(dummy_input)
    
    assert torch.isfinite(output).all(), "Model output contains NaN/Inf on test data."
    
    # Check if weights are not all zeros (trivial solution)
    param_sum = sum([p.abs().sum() for p in model.parameters()])
    assert param_sum > 0.1, "Model weights have collapsed to zero. This is not a valid fix."