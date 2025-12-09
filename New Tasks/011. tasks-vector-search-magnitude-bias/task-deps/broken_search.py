import numpy as np
import os
import json

DATA_PATH = "/app/data/embeddings.npz"
OUTPUT_PATH = "/app/output/results.json"

def search():
    print("Loading Index...")
    data = np.load(DATA_PATH)
    docs = data['docs']          # Shape: (10000, 128)
    queries = data['queries']    # Shape: (100, 128)
    ground_truth = data['ground_truth']
    
    print(f"Index loaded. Docs: {docs.shape}")
    
    # --- BROKEN SEARCH ENGINE LOGIC ---
    
    # PROBLEM: The engineer assumes Dot Product == Similarity.
    # But 'docs' have wildly different magnitudes.
    # A vector with length 50 will act like a magnet, attracting all queries
    # even if it points in the wrong direction.
    
    print("Running Nearest Neighbor Search (Inner Product)...")
    # scores[i, j] = dot(query[i], doc[j])
    scores = np.dot(queries, docs.T)
    
    # Get Top-1 result for each query
    # argsort returns indices of sorted elements. [:, ::-1] reverses to descending order.
    top_indices = np.argsort(scores, axis=1)[:, ::-1]
    top_1 = top_indices[:, 0]
    
    # ----------------------------------
    
    # Calculate Metric
    matches = (top_1 == ground_truth)
    recall = np.mean(matches)
    
    print(f"Search Complete.")
    print(f"Recall@1: {recall:.2%}")
    print("---------------------------------------------------")
    if recall < 0.1:
        print("ALERT: Recall is near zero. High magnitude vectors are dominating results.")
        print("HINT: You need to implement Cosine Similarity (normalize vectors).")
    
    # Save Results
    results = {
        "recall": float(recall),
        "predictions": top_1.tolist()
    }
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    search()