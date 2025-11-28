import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

OUTPUT_DIR = "/app/data"

def generate_feature_store_data():
    print("Generating Feature Store Data...")
    np.random.seed(42)
    
    n_users = 100
    n_updates_per_user = 20
    n_applications = 1000
    
    # 1. Generate Credit Score History (The Feature)
    # A log of updates: User X changed score to Y at Time T
    history_records = []
    start_date = datetime(2024, 1, 1)
    
    for uid in range(n_users):
        curr_time = start_date
        # Base score
        score = np.random.randint(300, 850)
        
        for _ in range(n_updates_per_user):
            # Time jumps forward 1-10 days
            curr_time += timedelta(days=np.random.randint(1, 10))
            # Score drifts
            score += np.random.randint(-50, 50)
            score = np.clip(score, 300, 850)
            
            history_records.append({
                "user_id": uid,
                "credit_score": score,
                "score_updated_at": curr_time
            })
            
    df_history = pd.DataFrame(history_records)
    
    # 2. Generate Loan Applications (The Label)
    # Applications happen randomly within the timeline
    app_records = []
    timeline_end = curr_time
    
    for i in range(n_applications):
        uid = np.random.randint(0, n_users)
        # Random time in the year
        days_offset = np.random.randint(0, 365)
        app_time = start_date + timedelta(days=days_offset)
        
        # Ground Truth Logic (Hidden from Agent in this file)
        # Random approval based on nothing (we care about the join, not the ML quality here)
        is_approved = np.random.choice([0, 1])
        
        app_records.append({
            "application_id": f"APP_{i:05d}",
            "user_id": uid,
            "application_at": app_time,
            "is_approved": is_approved
        })
        
    df_apps = pd.DataFrame(app_records)
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_history.to_parquet(os.path.join(OUTPUT_DIR, "credit_score_history.parquet"))
    df_apps.to_parquet(os.path.join(OUTPUT_DIR, "loan_applications.parquet"))
    
    print("Data Generation Complete.")

if __name__ == "__main__":
    generate_feature_store_data()