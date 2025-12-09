import pandas as pd
import numpy as np

def generate_features(df):
    """
    Generates technical indicators.
    Input: DataFrame with 'timestamp', 'price', 'volume'.
    """
    # Ensure sorted by time
    df = df.sort_values('timestamp').set_index('timestamp')

    # ---------------------------------------------------------
    # BUG 1: Windowing Logic
    # 'center=True' peeks into the future.
    # ---------------------------------------------------------
    df['volatility_60s'] = df['price'].rolling(window=60, center=True).std()

    # ---------------------------------------------------------
    # BUG 2: Global Statistics Leakage
    # Using global mean/std normalizes the past using future knowledge.
    # ---------------------------------------------------------
    global_mean = df['price'].mean()
    global_std = df['price'].std()
    df['zscore_price_60s'] = (df['price'] - global_mean) / global_std

    # ---------------------------------------------------------
    # BUG 3: Row-Count vs Time-Based
    # Shift(60) is 60 rows, not 60 seconds.
    # ---------------------------------------------------------
    df['momentum_60s'] = df['price'] - df['price'].shift(60)

    # Drop NaNs created by windowing
    df = df.dropna()
    
    return df