import numpy as np
import os
import json
import sys

# Import agent's code
sys.path.append("/app/src")
from sanitizer import GradientSanitizer

DATA_PATH = "/app/data/batch_gradients.npy"
OUTPUT_TOKEN = "/app/output/token.json"

def audit():
    print("Running Privacy Audit...")
    
    # 1. Load Data
    grads = np.load(DATA_PATH) # Shape (32, 100)
    batch_size = grads.shape[0]
    
    # 2. Baseline Run (Clean Data)
    sanitizer = GradientSanitizer(clip_norm=1.0, noise_multiplier=0.0) # Disable noise to test clipping logic deterministically
    baseline_agg = sanitizer.sanitize_gradients(grads)
    
    # 3. Attack Run (Inject Canary)
    # We take the first row and multiply it by 1,000,000
    # This simulates one user having EXTREME data.
    attack_grads = grads.copy()
    outlier_scale = 1000000.0
    attack_grads[0] = attack_grads[0] * outlier_scale
    
    attack_agg = sanitizer.sanitize_gradients(attack_grads)
    
    # 4. Measure Influence
    # Influence = Euclidean distance between clean aggregation and attacked aggregation
    influence = np.linalg.norm(attack_agg - baseline_agg)
    
    print(f"Baseline Norm: {np.linalg.norm(baseline_agg):.4f}")
    print(f"Attack Norm:   {np.linalg.norm(attack_agg):.4f}")
    print(f"Canary Influence (Shift): {influence:.4f}")
    
    # 5. The Criterion
    # If Per-Sample Clipping is used:
    # The outlier row would be clipped to max_norm (1.0).
    # Its contribution to the mean changes from ~10.0/N to 1.0/N.
    # The shift should be very small (approx 1.0/32 ~ 0.03).
    
    # If Batch Clipping is used:
    # The mean vector points purely in the direction of the outlier.
    # The shift is massive (limited only by the final clip_norm).
    # Influence will be close to 1.0 or higher depending on geometry.
    
    if influence > 0.5:
        print("\n[FAIL] Audit Failed: Single user altered the global gradient significantly.")
        print("Diagnosis: You are using Batch Clipping. Switch to Per-Sample Clipping.")
        sys.exit(1)
    else:
        print("\n[PASS] Audit Passed: The outlier's influence was bounded.")
        
        os.makedirs(os.path.dirname(OUTPUT_TOKEN), exist_ok=True)
        with open(OUTPUT_TOKEN, "w") as f:
            json.dump({"status": "verified", "influence": influence}, f)

if __name__ == "__main__":
    audit()