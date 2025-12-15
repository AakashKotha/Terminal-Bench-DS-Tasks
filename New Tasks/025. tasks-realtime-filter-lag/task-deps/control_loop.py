import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, lfilter
import os

INPUT_PATH = "/app/data/sensor_input.csv"
OUTPUT_PATH = "/app/output/smoothed_sensor.csv"

def process_sensor_data():
    if not os.path.exists(INPUT_PATH):
        return
        
    df = pd.read_csv(INPUT_PATH)
    raw = df["raw_value"].values
    
    # --- BROKEN LOGIC START ---
    
    # Design a low-pass Butterworth filter
    # Wn = 0.05 is the cutoff frequency
    b, a = butter(N=3, Wn=0.05, btype='low')
    
    # ERROR: 'filtfilt' applies the filter forward and then backward.
    # This cancels out phase delay (Lag = 0), which is great!
    # BUT, to filter backwards, you need the END of the array.
    # In real-time, we don't have the end of the array yet.
    # This simulation mimics "Offline Processing" which misled the engineers.
    
    smoothed = filtfilt(b, a, raw)
    
    # --- BROKEN LOGIC END ---
    
    df["smoothed_value"] = smoothed
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved smoothed data to {OUTPUT_PATH}")

if __name__ == "__main__":
    process_sensor_data()