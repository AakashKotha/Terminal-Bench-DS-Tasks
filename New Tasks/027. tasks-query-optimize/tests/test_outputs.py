import sqlite3
import pytest
import os
import time

DB_PATH = "/app/db/analytics.db"
SOLUTION_SQL = "/app/solution.sql"
BASELINE_SQL = "/app/task-deps/my-sql-query.sql"
GOLDEN_SQL_PATH = "tests/golden.sql" # Helper to store the 'correct' answer logic for validation

def test_files_exist():
    assert os.path.exists(SOLUTION_SQL), "solution.sql not found."
    assert os.path.exists(DB_PATH), "Database not found."

def test_query_correctness():
    """
    Does the optimized query return the EXACT same number as the baseline logic?
    """
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Run Baseline (We assume we can run it, it might be slow but 1M rows is acceptable for a single run in test)
    # To save test time, we can pre-calculate the truth or run a trusted efficient version.
    # Let's run a trusted efficient version to get the Ground Truth count.
    
    start_truth = "2024-01-01 00:00:00"
    end_truth = "2024-02-01 00:00:00"
    
    ground_truth_query = f"""
    SELECT count(*) 
    FROM logs l
    JOIN users u ON l.user_id = u.id
    WHERE u.country = 'US'
      AND l.event_type = 'LOGIN'
      AND l.timestamp >= '{start_truth}' 
      AND l.timestamp < '{end_truth}'
    """
    
    cursor = conn.cursor()
    cursor.execute(ground_truth_query)
    expected_count = cursor.fetchone()[0]
    
    # 2. Run Agent Solution
    with open(SOLUTION_SQL, 'r') as f:
        agent_query = f.read()
        
    start_time = time.time()
    cursor.execute(agent_query)
    agent_count = cursor.fetchone()[0]
    duration = time.time() - start_time
    
    conn.close()
    
    print(f"Expected: {expected_count}, Got: {agent_count}")
    print(f"Execution Time: {duration:.4f}s")
    
    assert agent_count == expected_count, "Your query returned the wrong result count."

def test_performance_benchmark():
    """
    Is it fast? < 200ms.
    This implies they added indices.
    """
    conn = sqlite3.connect(DB_PATH)
    
    with open(SOLUTION_SQL, 'r') as f:
        agent_query = f.read()
        
    # Warmup
    conn.execute(agent_query)
    
    # Measure
    start_time = time.time()
    for _ in range(5): # Average of 5 runs
        conn.execute(agent_query)
    avg_time = (time.time() - start_time) / 5.0
    
    conn.close()
    
    print(f"Average Execution Time: {avg_time:.4f}s")
    
    # Without index: ~0.5s to 1.0s on 1M rows (SQLite is fast, but full scan is measurable)
    # With index: ~0.01s
    
    # We set threshold at 0.15s to be safe against noisy runners, 
    # but strictly lower than full scan time.
    assert avg_time < 0.2, \
        f"Query too slow ({avg_time:.4f}s). Did you create the necessary indices on 'user_id' and 'timestamp'?"

def test_indices_created():
    """
    Directly check schema for indices.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check logs indices
    cursor.execute("PRAGMA index_list(logs)")
    indices = cursor.fetchall()
    # indices structure: (seq, name, unique, origin, partial)
    
    index_names = [i[1] for i in indices]
    print(f"Indices on 'logs': {index_names}")
    
    # We expect at least one index that covers user_id or timestamp
    assert len(indices) > 0, "No indices found on 'logs' table."
    
    conn.close()