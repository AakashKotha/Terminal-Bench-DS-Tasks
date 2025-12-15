import numpy as np
import pandas as pd
import os

OUTPUT_DIR = "/app/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_sensor_data():
    print("Generating Noisy Sensor Data...")
    np.random.seed(42)
    
    n_steps = 200
    
    # 1. Ground Truth Signal (Step Function)
    # 0-50: Altitude 0
    # 50-200: Altitude 10
    truth = np.zeros(n_steps)
    truth[50:] = 10.0
    
    # 2. Add Noise
    # Sigma = 1.0. Variance = 1.0.
    noise = np.random.normal(0, 1.0, n_steps)
    
    sensor_reading = truth + noise
    
    # Save
    df = pd.DataFrame({
        "time": np.arange(n_steps),
        "raw_value": sensor_reading,
        "ground_truth": truth
    })
    
    df.to_csv(os.path.join(OUTPUT_DIR, "sensor_input.csv"), index=False)
    print("Data generated.")

if __name__ == "__main__":
    generate_sensor_data()