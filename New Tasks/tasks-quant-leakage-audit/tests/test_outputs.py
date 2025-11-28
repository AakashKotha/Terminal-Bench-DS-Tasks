import sys
import os
import pytest
import pandas as pd
import numpy as np

# Ensure we can import the agent's code
sys.path.append("/app/src")

# Input Path
INPUT_PATH = "/app/data/tick_data.parquet"

def test_module_structure():
    """Validates that the agent modified the correct file and function exists."""
    assert os.path.exists("/app/src/feature_pipeline.py"), "File /app/src/feature_pipeline.py not found."
    
    try:
        from feature_pipeline import generate_features
    except ImportError:
        pytest.fail("Could not import 'generate_features' from 'feature_pipeline'. Check function name.")

def test_causality_probe_global_stats():
    """
    Test for Global Statistics Leakage (Bug 2).
    We add a massive price jump in the FUTURE.
    If the PAST Z-scores change, the agent is still using global stats.
    """
    from feature_pipeline import generate_features
    
    # Load original data
    df_orig = pd.read_parquet(INPUT_PATH).sort_values("timestamp")
    
    # Create "Shocked" Scenario
    split_idx = len(df_orig) // 2
    df_shock = df_orig.copy()
    
    # Add $1,000,000 to all prices in the SECOND HALF of the data
    shock_col_idx = df_shock.columns.get_loc('price')
    df_shock.iloc[split_idx:, shock_col_idx] += 1000000.0
    
    # Run Agent's Pipeline
    try:
        res_orig = generate_features(df_orig.copy())
        res_shock = generate_features(df_shock.copy())
    except Exception as e:
        pytest.fail(f"Pipeline crashed during execution: {e}")
    
    # CHECK THE PAST (A point well before the shock)
    test_idx = split_idx - 100
    ts_check = df_orig.iloc[test_idx]['timestamp']
    
    # Get the Z-Score at this past timestamp for both scenarios
    # We use a try/except in case the agent dropped indices or renamed columns differently
    try:
        val_orig = res_orig.loc[ts_check]['zscore_price_60s']
        val_shock = res_shock.loc[ts_check]['zscore_price_60s']
    except KeyError:
        pytest.fail("Output missing required column 'zscore_price_60s' or timestamp index is broken.")

    # The Z-score in the past should be identicial, regardless of the future price explosion.
    # If they use global mean, val_shock will be tiny because the mean exploded.
    assert np.isclose(val_orig, val_shock, atol=1e-5), \
        f"Global Leakage Detected! Future prices altered past Z-Scores.\n" \
        f"Original Z: {val_orig}, Shocked Z: {val_shock}.\n" \
        f"Hint: Use a rolling window for Z-score (mean/std), do not use df['price'].mean()."

def test_causality_probe_window_peeking():
    """
    Test for Window Lookahead (Bug 1).
    We modify the VERY LAST data point.
    If 'volatility' at (Last Point - 1) changes, the window was centered (peeking).
    """
    from feature_pipeline import generate_features
    
    df_orig = pd.read_parquet(INPUT_PATH).sort_values("timestamp")
    df_shock = df_orig.copy()
    
    # Shock the LAST point
    last_idx = len(df_shock) - 1
    shock_col_idx = df_shock.columns.get_loc('price')
    df_shock.iloc[last_idx, shock_col_idx] += 5000.0 
    
    res_orig = generate_features(df_orig.copy())
    res_shock = generate_features(df_shock.copy())
    
    # Check a point just before the end.
    # If window is centered (e.g. window=60, center=True), T-1 sees T.
    # If window is trailing (causal), T-1 ONLY sees T-1, T-2...
    
    test_idx = last_idx - 1
    ts_check = df_orig.iloc[test_idx]['timestamp']
    
    val_orig = res_orig.loc[ts_check]['volatility_60s']
    val_shock = res_shock.loc[ts_check]['volatility_60s']
    
    assert np.isclose(val_orig, val_shock, atol=1e-5), \
        f"Look-Ahead Bias Detected! Changing the last price changed the previous volatility.\n" \
        f"Hint: Ensure rolling windows are NOT centered."

def test_time_based_window_enforcement():
    """
    Test for Row vs Time Windows (Bug 3).
    We create synthetic data with a huge time gap.
    """
    from feature_pipeline import generate_features
    
    # T1: 09:00:00, Price 100
    # T2: 09:00:01, Price 100
    # ... 5 minute gap ...
    # T3: 09:05:00, Price 200
    
    df_gap = pd.DataFrame({
        "timestamp": [
            pd.Timestamp("2024-01-01 09:00:00"),
            pd.Timestamp("2024-01-01 09:00:01"),
            pd.Timestamp("2024-01-01 09:05:00")
        ],
        "price": [100.0, 100.0, 200.0],
        "volume": [100, 100, 100]
    })
    
    res = generate_features(df_gap)
    
    # Check T3.
    # If window="60s", T3 only sees T3 (gap > 60s). Volatility of 1 point is NaN or 0.
    # If window=60 (rows), T3 sees T2 and T1. Volatility of [100, 100, 200] is high (~57).
    
    ts_t3 = pd.Timestamp("2024-01-01 09:05:00")
    
    if ts_t3 not in res.index:
        # Agent might have dropped it due to NaNs, which is acceptable if strict
        return

    vol_t3 = res.loc[ts_t3]['volatility_60s']
    
    # If NaN, it passed (pandas rolling std of 1 item is NaN)
    if pd.isna(vol_t3):
        return

    assert vol_t3 < 10.0, \
        f"Row-Based Window Detected. Volatility across a 5-minute gap was calculated as {vol_t3}.\n" \
        f"It should be effectively 0 or NaN because the window is '60s'."