import json
import numpy as np
import pandas as pd
import pytest
import os

OUTPUT_PATH = "/app/output/recommendations.json"
DATA_PATH = "/app/data/interactions.parquet"

# Thresholds defined in Task
MIN_COVERAGE = 0.15 # 15%
MIN_NDCG = 0.10     # 0.10

@pytest.fixture(scope="module")
def recommendations():
    assert os.path.exists(OUTPUT_PATH), "Output file /app/output/recommendations.json not found."
    with open(OUTPUT_PATH, 'r') as f:
        return json.load(f)

@pytest.fixture(scope="module")
def ground_truth():
    # Load original interactions to validate relevance
    return pd.read_parquet(DATA_PATH)

def calculate_ndcg(recs, ground_truth_df, k=10):
    # Simplified NDCG calculation on Training Data (Reconstruction)
    # A recommendation is "Relevant" (1) if the user actually interacted with it in the source data
    # In a real scenario, we'd use a holdout set, but reconstruction error is fine for this benchmark.
    
    user_truth = ground_truth_df.groupby('user_id')['item_id'].apply(set).to_dict()
    
    ndcg_scores = []
    
    for uid_str, items in recs.items():
        uid = int(uid_str)
        if uid not in user_truth:
            continue
            
        true_items = user_truth[uid]
        dcg = 0.0
        idcg = 0.0
        
        # Calculate DCG
        for i, item in enumerate(items[:k]):
            rel = 1 if item in true_items else 0
            dcg += rel / np.log2(i + 2)
            
        # Calculate IDCG (Ideal case: all relevant items at the top)
        n_relevant = min(len(true_items), k)
        for i in range(n_relevant):
            idcg += 1 / np.log2(i + 2)
            
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)
            
    return np.mean(ndcg_scores)

def test_catalog_coverage(recommendations, ground_truth):
    """
    Test 1: Diversity (The Hard Part).
    Did the agent recommend enough unique items across the population?
    """
    n_total_items = ground_truth['item_id'].max() + 1
    
    # Collect all unique recommended items
    recommended_set = set()
    for u, items in recommendations.items():
        recommended_set.update(items)
        
    coverage = len(recommended_set) / n_total_items
    print(f"\n\n--- METRICS REPORT ---")
    print(f"Unique Items Recommended: {len(recommended_set)} / {n_total_items}")
    print(f"Catalog Coverage: {coverage:.2%}")
    
    assert coverage >= MIN_COVERAGE, \
        f"FAILED: Catalog Coverage is too low ({coverage:.2%}). " \
        f"Required: > {MIN_COVERAGE:.0%}. " \
        "Your model is only recommending popular items (Harry Potter Effect). " \
        "Implement a re-ranking strategy (e.g., MMR, penalty)."

def test_relevance_ndcg(recommendations, ground_truth):
    """
    Test 2: Accuracy/Relevance.
    Did the agent simply pick random items to cheat coverage?
    The recommendations must still be related to user preferences.
    """
    ndcg = calculate_ndcg(recommendations, ground_truth)
    print(f"NDCG@10: {ndcg:.4f}")
    print("----------------------\n")
    
    assert ndcg >= MIN_NDCG, \
        f"FAILED: NDCG is too low ({ndcg:.4f}). " \
        f"Required: > {MIN_NDCG}. " \
        "You diversified too aggressively and lost relevance. " \
        "Balance the trade-off."

def test_structure_validity(recommendations):
    """Basic format check."""
    assert len(recommendations) > 0
    # Check first user
    first_key = next(iter(recommendations))
    assert isinstance(recommendations[first_key], list)
    assert len(recommendations[first_key]) == 10, "Must recommend exactly 10 items per user."