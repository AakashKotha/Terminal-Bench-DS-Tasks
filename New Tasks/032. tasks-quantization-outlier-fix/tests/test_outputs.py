import torch
import pytest
import sys
import os

# Ensure we can import from task-deps
sys.path.append(os.path.join(os.getcwd(), 'task-deps'))

from model import SimpleClassifier
from data import generate_clean_data, generate_calibration_data_with_outliers
from quantizer import Quantizer

def get_accuracy(model, quantizer, X, y):
    """
    Evaluates model accuracy using the quantizer for the input layer.
    """
    model.eval()
    with torch.no_grad():
        # Pass input through fake quantization
        X_quantized = quantizer.quantize_forward(X)
        
        # Pass through model
        logits = model(X_quantized)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean().item()
    return acc

def test_quantization_robustness():
    print("\n--- Starting Quantization Robustness Test ---")
    
    # 1. Setup Model and Data
    model = SimpleClassifier()
    # We don't train the model here to save time/compute. 
    # We rely on the fact that an uninitialized model with random weights 
    # acts as a consistent hash function. If inputs are preserved, 
    # outputs remain consistent. If inputs are zeroed out (due to bad quantization),
    # outputs effectively become random/garbage relative to the "float" baseline behavior.
    # However, to make the test strictly about signal preservation, let's look at 
    # Reconstruction Error (MSE) of the input signal, or check simple linear separability.
    
    # Actually, let's verify signal preservation directly to be robust to random weights.
    
    # Generate Calibration Data (Dirty)
    calib_data = generate_calibration_data_with_outliers(n_samples=500)
    # Convert to a simple loader list
    calib_loader = [calib_data[i:i+50] for i in range(0, 500, 50)]
    
    # 2. Run User's Calibration Logic
    quantizer = Quantizer(model)
    quantizer.calibrate(calib_loader)
    
    # 3. Verify Range Logic
    # The clean data is roughly [-4, 4]. Outliers are +/- 500.
    # If logic is fixed (percentile), min/max should be close to -4/4.
    # If logic is broken (minmax), range will be huge (approx 500 or 1000).
    
    print(f"Test Observed - Min: {quantizer.min_val}, Max: {quantizer.max_val}")
    
    range_width = quantizer.max_val - quantizer.min_val
    
    if range_width > 100:
        pytest.fail(
            f"Quantization range is too wide ({range_width:.2f}). "
            "It seems outliers are determining the scale. "
            "Did you implement percentile/robust calibration in `update_stats`?"
        )
        
    # 4. Verify Reconstruction Quality on Clean Data
    clean_X, _ = generate_clean_data(100)
    reconstructed_X = quantizer.quantize_forward(clean_X)
    
    mse = torch.mean((clean_X - reconstructed_X) ** 2).item()
    print(f"Mean Squared Error (Float vs Int8): {mse:.4f}")
    
    # A good quantization (range ~8, 256 bins) should have MSE < 0.01 approx.
    # A bad quantization (range ~1000, 256 bins -> step ~4) implies error ~2.0.
    if mse > 0.5:
        pytest.fail(f"Reconstruction MSE is too high ({mse:.4f}). Precision lost.")
        
    print("Test Passed: Quantizer ignored outliers and preserved signal precision.")