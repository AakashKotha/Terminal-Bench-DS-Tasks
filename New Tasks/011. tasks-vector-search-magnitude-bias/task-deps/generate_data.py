import numpy as np
import os

OUTPUT_DIR = "/app/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_vectors():
    print("Generating Embedding Data...")
    np.random.seed(42)
    
    dim = 128
    n_docs = 10000
    n_queries = 100
    
    # 1. Generate Base Vectors (Random Unit Spheres)
    # These represent the "Semantic Directions"
    docs_base = np.random.randn(n_docs, dim)
    docs_base /= np.linalg.norm(docs_base, axis=1, keepdims=True)
    
    queries = np.random.randn(n_queries, dim)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    
    # 2. Inject Ground Truth (The "Needles")
    # For each query, we pick one document index to be the "Perfect Match"
    # We set its direction to be very close to the query (High Cosine Similarity)
    # BUT we set its MAGNITUDE to be SMALL (e.g., 1.0)
    ground_truth_ids = []
    
    for i in range(n_queries):
        target_doc_idx = i  # Simple mapping: Query 0 -> Doc 0 is the truth
        ground_truth_ids.append(target_doc_idx)
        
        # Make Doc match Query direction almost exactly
        # Cosine Sim ~ 1.0
        docs_base[target_doc_idx] = queries[i] + np.random.normal(0, 0.01, dim)
        
    # 3. Apply The "Magnitude Trap"
    # We scale the vectors.
    # - The "True Matches" get Unit Length (1.0)
    # - The "Random Noise" docs get MASSIVE Length (e.g., 50.0)
    
    # By default, dot product is |A||B|cos(theta).
    # Truth: 1.0 * 1.0 * cos(0) = 1.0
    # Noise: 1.0 * 50.0 * cos(85deg) = 50 * 0.08 = 4.0
    # Result: Noise score (4.0) > Truth score (1.0). Search fails.
    
    doc_magnitudes = np.random.uniform(20.0, 50.0, size=(n_docs, 1))
    
    # Fix magnitudes for ground truth to be small (1.0)
    for idx in ground_truth_ids:
        doc_magnitudes[idx] = 1.0
        
    final_docs = docs_base * doc_magnitudes
    
    # 4. Save
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "embeddings.npz"),
        docs=final_docs,
        queries=queries,
        ground_truth=np.array(ground_truth_ids)
    )
    
    print(f"Generated {n_docs} docs and {n_queries} queries.")
    print("Trap Set: Irrelevant docs have 20x-50x larger magnitude than relevant docs.")

if __name__ == "__main__":
    generate_vectors()