import torch
import torch.nn.functional as F
import pytest
import sys
import os

# Add src to path
sys.path.append("/app/src")

from train_distillation import compute_distillation_loss
from models import StudentModel

def test_distillation_math_logic():
    """
    We verify the custom loss function specifically.
    We pass in dummy logits and check if the output matches the CORRECT formula.
    """
    # 1. Setup
    T = 4.0
    alpha = 1.0 # Pure distillation to isolate the bug
    
    # Batch size 1, 2 classes
    s_logits = torch.tensor([[10.0, 5.0]], requires_grad=True) # Strong prediction class 0
    t_logits = torch.tensor([[10.0, 5.0]]) # Teacher agrees
    labels = torch.tensor([0]) # Not used if alpha=1.0
    
    # 2. Compute Agent's Loss
    agent_loss = compute_distillation_loss(s_logits, t_logits, labels, T=T, alpha=alpha)
    
    # 3. Compute Correct Loss Manually
    # Correct: T^2 * KL( log_softmax(S/T), softmax(T/T) )
    
    inp = F.log_softmax(s_logits / T, dim=1)
    target = F.softmax(t_logits / T, dim=1)
    kl = F.kl_div(inp, target, reduction="batchmean")
    expected_loss = (T ** 2) * kl
    
    # 4. Compare
    print(f"Agent Loss: {agent_loss.item()}")
    print(f"Expected Loss: {expected_loss.item()}")
    
    # Tolerance allows for minor implementation details, but a missing T^2 (factor of 16) 
    # or incorrect log_softmax usage will cause massive divergence.
    assert torch.isclose(agent_loss, expected_loss, rtol=0.1), \
        f"Loss calculation incorrect. Expected ~{expected_loss.item():.4f}, got {agent_loss.item():.4f}. " \
        "Did you scale by T^2? Did you use log_softmax for input?"

def test_student_accuracy():
    """
    Checks if the saved model actually works on the dataset.
    """
    # Load Data
    data_path = "/app/data/dataset.pt"
    if not os.path.exists(data_path):
        pytest.fail("Data file missing.")
        
    data = torch.load(data_path)
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    # Load Model
    model_path = "/app/output/student.pth"
    if not os.path.exists(model_path):
        pytest.fail("Student model artifact not found.")
        
    student = StudentModel()
    student.load_state_dict(torch.load(model_path))
    student.eval()
    
    with torch.no_grad():
        preds = student(X_test).argmax(1)
        acc = (preds == y_test).float().mean().item()
        
    print(f"Verified Test Accuracy: {acc:.2%}")
    
    # 85% is a strong threshold that requires the T^2 scaling to hit
    # Without T^2, gradients are too small, student underfits.
    assert acc > 0.85, f"Student accuracy ({acc:.2%}) is below the success threshold (85%). Distillation failed."