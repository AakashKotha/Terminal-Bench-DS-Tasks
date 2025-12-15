import json
import pytest
import os

OUTPUT_FILE = "/app/output/recommendations.json"
METADATA_FILE = "/app/data/metadata.json"

def test_results_exist():
    assert os.path.exists(OUTPUT_FILE), "Output file recommendations.json not found."

def test_semantic_relevance():
    """
    We verify that the agent's search returned items from the 'Nature' category.
    The broken implementation returns 'Action' because they have 100x magnitude.
    The correct implementation (Cosine Sim) returns 'Nature'.
    """
    with open(OUTPUT_FILE, 'r') as f:
        result_ids = json.load(f)
        
    with open(METADATA_FILE, 'r') as f:
        meta = json.load(f)
        
    # Create lookup
    meta_map = {m['id']: m for m in meta}
    
    categories = []
    for rid in result_ids:
        assert rid in meta_map, f"Invalid ID {rid} returned."
        categories.append(meta_map[rid]['category'])
        
    print(f"Agent returned categories: {categories}")
    
    # Strict check: All top 3 must be Nature
    # If even one is "Action", it implies magnitude bias wasn't fixed.
    
    action_count = categories.count("Action")
    nature_count = categories.count("Nature")
    
    if action_count > 0:
        pytest.fail(f"Search Results biased! Found {action_count} 'Action' movies (High Magnitude) for a Nature query. Did you normalize the vectors?")
        
    assert nature_count == 3, f"Expected 3 Nature documentaries, found {nature_count}. Results: {categories}"
