import numpy as np
import json
import os

DATA_DIR = "/app/data"
OUTPUT_FILE = "/app/output/recommendations.json"

def load_data():
    emb = np.load(os.path.join(DATA_DIR, "embeddings.npy"))
    query = np.load(os.path.join(DATA_DIR, "query.npy"))
    with open(os.path.join(DATA_DIR, "metadata.json"), 'r') as f:
        meta = json.load(f)
    return emb, query, meta

def search(query_vector, database_vectors, top_k=3):
    """
    Performs vector search.
    Current Implementation: Dot Product.
    scores = db_vecs @ query
    """
    # --- BROKEN LOGIC START ---
    # This maximizes inner product.
    # Since 'database_vectors' have varying magnitudes (Action movies >> Nature docs),
    # the Action movies will win even if the angle is bad.
    
    scores = np.dot(database_vectors, query_vector)
    
    # --- BROKEN LOGIC END ---
    
    # Get top K indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices.tolist()

def main():
    db_vecs, query_vec, meta = load_data()
    
    print("Searching for: 'Nature Documentary'...")
    
    # Run search
    results_idx = search(query_vec, db_vecs)
    
    print("\n--- Search Results ---")
    results_data = []
    
    for idx in results_idx:
        item = meta[idx]
        print(f"ID: {item['id']} | Title: {item['title']} | Category: {item['category']} | Mag: {item['magnitude']:.1f}")
        results_data.append(item['id'])
        
    # Check if results look wrong (Self-Diagnosis)
    categories = [meta[i]['category'] for i in results_idx]
    if "Action" in categories:
        print("\n[WARNING] Search is returning Action movies for a Nature query!")
        print("Diagnosis: The search is biased by vector magnitude.")
    else:
        print("\n[SUCCESS] Search returned relevant results.")

    # Save output for grading
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results_data, f)

if __name__ == "__main__":
    main()