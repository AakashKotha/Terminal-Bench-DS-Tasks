import numpy as np
import pandas as pd
import scipy.sparse as sp
import os

OUTPUT_DIR = "/app/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_interaction_data():
    print("Generating skewed interaction data...")
    np.random.seed(42)
    
    n_users = 1000
    n_items = 500
    n_interactions = 20000
    
    # 1. Create Power Law Item Probabilities (The "Harry Potter" Effect)
    # Items 0-10 will get the vast majority of clicks
    item_probs = 1.0 / (np.arange(1, n_items + 1) ** 0.8)
    item_probs /= item_probs.sum()
    
    # 2. Create User Profiles (some prefer niche, most follow trends)
    user_ids = []
    item_ids = []
    ratings = []
    
    for u in range(n_users):
        # Number of interactions for this user
        n_user_inter = np.random.randint(5, 30)
        
        # Determine if user is "Niche" (10% chance) or "Mainstream"
        is_niche = np.random.random() < 0.10
        
        if is_niche:
            # Niche users pick from the tail
            # We simulate this by reversing probabilities or picking random
            u_probs = np.ones(n_items) / n_items
        else:
            # Mainstream users follow the power law
            u_probs = item_probs
            
        # Sample items
        chosen_items = np.random.choice(np.arange(n_items), size=n_user_inter, p=u_probs, replace=False)
        
        user_ids.extend([u] * n_user_inter)
        item_ids.extend(chosen_items)
        # Implicit ratings (all 1s)
        ratings.extend([1] * n_user_inter)
        
    # Save sparse matrix data
    df = pd.DataFrame({
        "user_id": user_ids,
        "item_id": item_ids,
        "rating": ratings
    })
    
    # Split Train/Test (Last interaction per user is test)
    # For simplicity in this task, we just provide the full training set 
    # and the agent must predict for these users.
    # The 'ground truth' for NDCG will be derived from the input itself (reconstruction) 
    # or a holdout set logic if we want to be stricter.
    # To keep it self-contained: We will evaluate NDCG on the Training Data itself 
    # (recommending items they actually liked) for the 'Relevance' metric, 
    # but the challenge is Diversity.
    
    df.to_parquet(os.path.join(OUTPUT_DIR, "interactions.parquet"))
    print(f"Data generated: {n_users} users, {n_items} items, {len(df)} interactions.")

if __name__ == "__main__":
    generate_interaction_data()