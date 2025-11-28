import pandas as pd
import numpy as np
import os

OUTPUT_DIR = "/app/data"

def generate_market_data():
    print("Generating market data...")
    np.random.seed(42)
    
    # 1. Generate irregular timestamps (Poisson process)
    # Average gap of 0.5 seconds, but some will be larger/smaller
    n_ticks = 10000
    time_gaps = np.random.exponential(scale=0.5, size=n_ticks)
    
    # Start from a fixed time
    start_time = pd.Timestamp("2024-01-01 09:30:00")
    timestamps = start_time + pd.to_timedelta(np.cumsum(time_gaps), unit="s")
    
    # 2. Generate Price Path (Geometric Brownian Motion)
    # Drift 0, Volatility 0.05% per tick
    returns = np.random.normal(loc=0.0, scale=0.0005, size=n_ticks)
    # Starting price $100
    price_path = 100 * np.exp(np.cumsum(returns))
    
    # 3. Generate Volume (Log-normal distribution)
    volumes = np.random.lognormal(mean=2.0, sigma=0.5, size=n_ticks) * 100
    volumes = volumes.astype(int)

    # 4. Create DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "price": price_path,
        "volume": volumes
    })

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save as Parquet (standard format for large datasets)
    output_path = os.path.join(OUTPUT_DIR, "tick_data.parquet")
    df.to_parquet(output_path)
    
    print(f"Success. Generated {len(df)} ticks of market data at {output_path}.")

if __name__ == "__main__":
    generate_market_data()