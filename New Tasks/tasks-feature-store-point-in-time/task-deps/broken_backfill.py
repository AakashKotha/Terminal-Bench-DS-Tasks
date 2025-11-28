import pandas as pd

def create_training_set():
    # Load Data
    print("Loading data...")
    df_apps = pd.read_parquet("/app/data/loan_applications.parquet")
    df_scores = pd.read_parquet("/app/data/credit_score_history.parquet")
    
    # --- BROKEN LOGIC STARTS HERE ---
    
    # BUG: Naive Merge
    # This joins ALL history rows for a user to the application.
    # It does not respect the timeline.
    print("Merging data...")
    merged = pd.merge(df_apps, df_scores, on="user_id", how="left")
    
    # BUG: Sorting by update time and keeping the LAST one.
    # This essentially gives every application the user's MOST RECENT score
    # regardless of when the application happened.
    # If I apply in Jan, but my score updates in Dec, this code gives me the Dec score.
    merged = merged.sort_values("score_updated_at")
    final_df = merged.drop_duplicates(subset=["application_id"], keep="last")
    
    # --- BROKEN LOGIC ENDS HERE ---
    
    print(f"Saving {len(final_df)} rows...")
    final_df.to_parquet("/app/data/training_set_corrected.parquet")

if __name__ == "__main__":
    create_training_set()