import pytest
import shutil
import os
import sys

# Ensure we can import the solution
sys.path.append("/app")

# Try to import solution, otherwise fallback (will fail logic tests)
try:
    if os.path.exists("/app/solution.py"):
        from solution import LSMStore
        print("Testing solution.py")
    else:
        # For the initial run or if agent writes to source directly
        sys.path.append("/app/task-deps")
        from storage import LSMStore
        print("Testing storage.py")
except ImportError:
    pytest.fail("Could not import LSMStore. Ensure you save your code to /app/solution.py")

DB_DIR = "test_data_dir"

@pytest.fixture
def store():
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    # Low max size to force frequent flushes
    db = LSMStore(data_dir=DB_DIR, max_memtable_size=10) 
    yield db
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

def test_data_resurrection_bug(store):
    """
    Scenario:
    1. Write Key A = 100. Flush to SST1.
    2. Delete Key A. Flush to SST2.
    3. At this point, Get(A) should return None (Memtable check or SST scan finds tombstone).
    4. Run Compact().
    5. Get(A).
       - Correct: Returns None (Tombstone persisted, or Key completely removed and not found).
       - Bug: Returns 100. (Tombstone was skipped, falling back to older insert).
    """
    
    # 1. Insert Data (SSTable 1)
    store.put("user_1", "active")
    store.flush()
    
    # Verify insert
    assert store.get("user_1") == "active", "Basic put/get failed"
    
    # 2. Delete Data (SSTable 2)
    store.delete("user_1")
    store.flush()
    
    # Verify delete pre-compaction (relies on read path logic working correctly)
    val = store.get("user_1")
    assert val is None, f"Pre-compaction delete failed. Expected None, got {val}"
    
    # 3. Compact
    # This should merge SST1 and SST2.
    # SST2 has ("user_1", deleted, ts=2)
    # SST1 has ("user_1", "active", ts=1)
    # Merge should see TS=2 is newest. 
    # If it ignores TS=2 because it's a delete, it might fall through to TS=1.
    print("Running Compaction...")
    store.compact()
    
    # 4. Verify Post-Compaction
    val_after = store.get("user_1")
    
    if val_after == "active":
        pytest.fail("FAIL: Data Resurrection Detected! 'user_1' reappeared after compaction.")
    
    assert val_after is None, f"Expected None, got {val_after}"
    print("Test Passed: Deleted data remained deleted after compaction.")

def test_update_precedence(store):
    """
    Ensure updates work correctly too.
    1. Put A=1. Flush.
    2. Put A=2. Flush.
    3. Compact.
    4. Get A should be 2.
    """
    store.put("price", "10")
    store.flush()
    
    store.put("price", "20")
    store.flush()
    
    store.compact()
    
    val = store.get("price")
    assert val == "20", f"Update precedence fail. Expected '20', got {val}. Did you accidentally keep the old record?"

def test_mixed_workload(store):
    """
    Complex chain.
    """
    store.put("k1", "v1")
    store.flush()
    store.delete("k1")
    store.flush()
    store.put("k1", "v2") # Resurrected by user, this is valid.
    store.flush()
    
    store.compact()
    assert store.get("k1") == "v2"