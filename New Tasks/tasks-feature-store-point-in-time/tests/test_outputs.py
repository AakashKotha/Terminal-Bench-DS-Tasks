import os
import pytest
import pandas as pd
import numpy as np

OUTPUT_PATH = "/app/data/training_set_corrected.parquet"
APPS_PATH = "/app/data/loan_applications.parquet"
SCORES_PATH = "/app/data/credit_score_history.parquet"

def test_file_exists():
    assert os.path.exists(OUTPUT_PATH), "Output file not found."

def test_schema_correctness():
    df = pd.read_parquet(OUTPUT_PATH)
    required_cols = [
        "application_id", "user_id", "application_at", 
        "credit_score", "score_updated_at"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

def test_no_future_leakage():
    """
    Check 1 (The Trap): 
    Are there any rows where score_updated_at > application_at?
    If yes, the agent failed to prevent leakage.
    """
    df = pd.read_parquet(OUTPUT_PATH)
    
    # Drop NaNs (cases where no history existed are fine)
    df = df.dropna(subset=["score_updated_at", "application_at"])
    
    # The crucial check
    leaked_rows = df[df["score_updated_at"] > df["application_at"]]
    
    assert len(leaked_rows) == 0, \
        f"Data Leakage Detected! Found {len(leaked_rows)} rows where the credit score comes from the FUTURE.\n" \
        f"Example: App at {leaked_rows.iloc[0]['application_at']}, but Score from {leaked_rows.iloc[0]['score_updated_at']}.\n" \
        "You must perform a Point-in-Time (As-Of) join."

def test_value_correctness_vs_ground_truth():
    """
    Check 2 (Ground Truth):
    We calculate the Golden Answer using pandas.merge_asof (the correct way).
    The agent's answer must match this EXACTLY.
    """
    # 1. Load Raw Data
    apps = pd.read_parquet(APPS_PATH).sort_values("application_at")
    scores = pd.read_parquet(SCORES_PATH).sort_values("score_updated_at")
    
    # 2. Compute Golden Set
    # merge_asof requires sorted keys
    golden = pd.merge_asof(
        apps, 
        scores, 
        left_on="application_at", 
        right_on="score_updated_at", 
        by="user_id", 
        direction="backward" # Strictly past or equal
    )
    
    # 3. Load Agent's Set
    agent = pd.read_parquet(OUTPUT_PATH)
    
    # 4. Join Agent to Golden on application_id to compare
    # We rename columns to avoid suffix hell
    comparison = pd.merge(
        golden[["application_id", "credit_score"]],
        agent[["application_id", "credit_score"]],
        on="application_id",
        suffixes=("_truth", "_agent"),
        how="inner"
    )
    
    # 5. Comparison
    # Handle NaNs: If truth is NaN (no history), agent must be NaN
    # If truth is Value, agent must be Value
    
    # Fill NaNs with -1 for easier comparison
    comparison = comparison.fillna(-1)
    
    matches = comparison["credit_score_truth"] == comparison["credit_score_agent"]
    accuracy = matches.mean()
    
    # We require 100% accuracy because this is a deterministic data operation
    assert accuracy == 1.0, \
        f"Value Mismatch! Your joined scores match the ground truth only {accuracy:.1%} of the time.\n" \
        "Ensure you are finding the *closest previous* timestamp for every user."