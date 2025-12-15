import pandas as pd
import numpy as np
import pytest
import os

OUTPUT_PATH = "/app/output/smoothed_sensor.csv"

def test_file_exists():
    assert os.path.exists(OUTPUT_PATH), "Output file not found."

def test_causality_check():
    """
    CRITICAL CHECK: Did the filter peek into the future?
    In the raw data, the step from 0 to 10 happens at index 50.
    
    A non-causal filter (filtfilt) will start rising BEFORE index 50 (e.g. at 45).
    A causal filter (lfilter) will start rising AT or AFTER index 50.
    """
    df = pd.read_csv(OUTPUT_PATH)
    smooth = df["smoothed_value"].values
    
    # Check values just before the step (indices 40-49)
    # The Ground Truth is 0.
    # If the filter sees the "10" coming at index 50, it will rise early.
    
    pre_step_mean = np.mean(smooth[40:50])
    
    # If it's significantly above 0 (e.g. > 1.0), it's likely non-causal pre-ringing
    # (filtfilt usually creates a symmetric ramp up)
    assert pre_step_mean < 1.0, \
        f"Non-Causal Behavior Detected! Signal started rising before the event occurred (Value={pre_step_mean:.2f}). " \
        "You must switch from 'filtfilt' to 'lfilter' or an online rolling mean."

def test_noise_reduction():
    """
    Did the agent smooth the signal enough?
    We inspect the 'steady state' region (indices 0-40) where truth=0.
    Variance should be low.
    """
    df = pd.read_csv(OUTPUT_PATH)
    smooth = df["smoothed_value"].values
    
    # Raw noise variance was ~1.0
    # We demand >90% reduction -> variance < 0.1
    # Or standard deviation < 0.32
    
    steady_state = smooth[10:45] # Avoid start transients and step
    variance = np.var(steady_state)
    
    print(f"Verified Noise Variance: {variance:.4f}")
    
    assert variance < 0.15, \
        f"Signal is too noisy (Var={variance:.4f}). Increase filter strength (lower cutoff or larger window)."

def test_latency_performance():
    """
    Did the agent make the filter TOO slow?
    Step happens at index 50 (Value goes 0 -> 10).
    We define 'Reaction Time' as when the signal crosses 5.0 (50%).
    """
    df = pd.read_csv(OUTPUT_PATH)
    smooth = df["smoothed_value"].values
    
    # Find first index where value > 5.0
    # Search after step start
    post_step = smooth[50:]
    crossing_indices = np.where(post_step > 5.0)[0]
    
    if len(crossing_indices) == 0:
        pytest.fail("Signal never reached target value (10.0). Filter might be broken.")
        
    # The index inside 'post_step' represents the lag
    lag = crossing_indices[0]
    
    print(f"Verified Lag: {lag} steps")
    
    # Constraint: Lag must be < 5 steps.
    # A heavy moving average (N=20) would have lag ~10.
    # A light EMA (alpha=0.5) has lag ~1-2.
    assert lag < 6, \
        f"Latency too high! Lag is {lag} steps. The drone will crash. " \
        "Tune your filter to respond faster (increase cutoff freq or reduce window size)."