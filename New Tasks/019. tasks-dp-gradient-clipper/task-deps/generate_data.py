import numpy as np
import os

OUTPUT_DIR = "/app/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_gradients():
    print("Generating Gradient Simulation Data...")
    np.random.seed(42)
    
    # Simulate a batch of gradients from a neural net layer
    # Batch Size = 32, Parameters = 100
    batch_size = 32
    params = 100
    
    # Normal gradients (from regular data)
    # Norms are roughly sqrt(100) ~ 10.0
    grads = np.random.normal(0, 1.0, (batch_size, params))
    
    # Save
    np.save(os.path.join(OUTPUT_DIR, "batch_gradients.npy"), grads)
    print(f"Saved gradients shape: {grads.shape}")

if __name__ == "__main__":
    generate_gradients()