import os
import json
import pytest
import pandas as pd

OUTPUT_PATH = "/app/output/clean_sessions.parquet"
LABELS_PATH = "/app/tests/hidden/labels.json"

def test_file_exists():
    assert os.path.exists(OUTPUT_PATH), "Output file not found."

def test_fraud_filtering_performance():
    """
    Validation Logic:
    1. Load the agent's cleaned dataframe.
    2. Load the hidden ground truth labels.
    3. Calculate Precision and Recall.
    
    Success Condition:
    - Must keep > 95% of Humans (Recall).
    - Must remove > 95% of Bots (Precision).
    
    This fails agents that:
    - Filter randomly.
    - Filter aggressively (deleting everyone).
    - Filter passively (keeping everyone).
    """
    # Load Truth
    with open(LABELS_PATH, 'r') as f:
        truth_map = json.load(f) # {session_id: 'human'/'bot'}
        
    # Load Submission
    try:
        df = pd.read_parquet(OUTPUT_PATH)
    except Exception as e:
        pytest.fail(f"Failed to read Parquet file: {e}")
        
    if "session_id" not in df.columns:
        pytest.fail("Output missing 'session_id' column.")
        
    submitted_ids = set(df["session_id"].unique())
    
    # Stats
    total_humans = sum(1 for v in truth_map.values() if v == 'human')
    total_bots = sum(1 for v in truth_map.values() if v == 'bot')
    
    humans_kept = 0
    bots_kept = 0
    
    for sid in submitted_ids:
        label = truth_map.get(sid)
        if label == "human":
            humans_kept += 1
        elif label == "bot":
            bots_kept += 1
            
    recall_humans = humans_kept / total_humans
    missed_bots = bots_kept / total_bots
    
    print(f"\nStats:")
    print(f"Humans Preserved: {humans_kept}/{total_humans} ({recall_humans:.1%})")
    print(f"Bots Leaked: {bots_kept}/{total_bots} ({missed_bots:.1%})")
    
    # 1. Did they delete everything?
    assert len(submitted_ids) > 100, "Output is too empty. You deleted almost everyone."
    
    # 2. Did they keep the humans? (Requires realizing Human timing is Variance > 0)
    assert recall_humans > 0.90, \
        f"You banned too many real users (Recall: {recall_humans:.1%}). Humans have irregular click times, don't ban them for variance!"
        
    # 3. Did they ban the bots? (Requires realizing Bots have Variance ~ 0 or Uniform Space)
    assert missed_bots < 0.10, \
        f"You failed to catch the bots (Leaked: {missed_bots:.1%}). Look for sessions with 'too perfect' timing or uniform spatial distribution."
