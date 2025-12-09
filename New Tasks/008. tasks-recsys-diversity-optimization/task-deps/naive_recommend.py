import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import json
import os

DATA_PATH = "/app/data/interactions.parquet"
OUTPUT_PATH = "/app/output/recommendations.json"
os.makedirs("/app/output", exist_ok=True)

def train_and_recommend():
    # 1. Load Data
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    
    n_users = df['user_id'].max() + 1
    n_items = df['item_id'].max() + 1
    
    # 2. Build Interaction Matrix
    # We use a simple pivot for demonstration (in prod, use scipy.sparse)
    # Filling 0 for unobserved
    print("Building matrix...")
    interaction_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
    
    # Ensure all columns/rows exist
    interaction_matrix = interaction_matrix.reindex(index=range(n_users), columns=range(n_items), fill_value=0)
    X = interaction_matrix.values
    
    # 3. Train SVD (Matrix Factorization)
    print("Training SVD...")
    svd = TruncatedSVD(n_components=20, random_state=42)
    user_factors = svd.fit_transform(X)
    item_factors = svd.components_
    
    # 4. Generate Recommendations
    print("Generating Recommendations (Naive)...")
    recommendations = {}
    
    # Reconstruct the full matrix of scores
    predicted_scores = np.dot(user_factors, item_factors)
    
    for u_id in range(n_users):
        # Get scores for this user
        scores = predicted_scores[u_id]
        
        # Mask items already interacted with (optional, but good practice)
        # known_positives = X[u_id] > 0
        # scores[known_positives] = -np.inf 
        
        # --- THE PROBLEM AREA ---
        # Greedy Selection: Just pick the top 10 highest scores.
        # Since popular items have higher magnitude vectors usually, they dominate.
        # The agent must change THIS logic.
        top_10_indices = np.argsort(scores)[::-1][:10]
        # ------------------------
        
        recommendations[int(u_id)] = [int(i) for i in top_10_indices]
        
    # 5. Save Output
    print(f"Saving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(recommendations, f)

if __name__ == "__main__":
    train_and_recommend()