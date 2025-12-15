import numpy as np
import json
import os

DATA_DIR = "/app/data"
os.makedirs(DATA_DIR, exist_ok=True)

def generate_embeddings():
    print("Generating Synthetic Embeddings...")
    np.random.seed(42)
    
    N_ITEMS = 1000
    DIM = 64
    
    # 3 Categories: 
    # 0: Action Blockbusters (High Magnitude)
    # 1: Nature Documentaries (Low Magnitude)
    # 2: RomComs (Medium Magnitude)
    
    categories = ["Action", "Nature", "RomCom"]
    
    # Random base directions for each category
    centroids = np.random.normal(0, 1, (3, DIM))
    # Normalize centroids
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    
    embeddings = []
    metadata = []
    
    for i in range(N_ITEMS):
        # Pick category
        cat_idx = np.random.choice([0, 1, 2])
        
        # Base vector = Centroid + Noise
        vec = centroids[cat_idx] + np.random.normal(0, 0.1, DIM)
        
        # Determine Magnitude based on category (The Bias Source)
        if cat_idx == 0: # Action - Popular!
            magnitude = np.random.uniform(50.0, 100.0)
            title = f"Explosion {i}"
        elif cat_idx == 1: # Nature - Niche
            magnitude = np.random.uniform(1.0, 5.0)
            title = f"Ants Life {i}"
        else:
            magnitude = np.random.uniform(10.0, 20.0)
            title = f"Love Story {i}"
            
        # Apply magnitude
        vec = vec / np.linalg.norm(vec) * magnitude
        
        embeddings.append(vec)
        metadata.append({
            "id": i,
            "title": title,
            "category": categories[cat_idx],
            "magnitude": float(magnitude) # Hint for debugging
        })
        
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # Save
    np.save(os.path.join(DATA_DIR, "embeddings.npy"), embeddings)
    with open(os.path.join(DATA_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f)
        
    # Save a Query Vector (A normalized Nature vector)
    # This represents the user searching for "Ants"
    query_vec = centroids[1] + np.random.normal(0, 0.05, DIM)
    query_vec = query_vec / np.linalg.norm(query_vec) # Query is unit length
    np.save(os.path.join(DATA_DIR, "query.npy"), query_vec)
    
    print("Data generated.")

if __name__ == "__main__":
    generate_embeddings()