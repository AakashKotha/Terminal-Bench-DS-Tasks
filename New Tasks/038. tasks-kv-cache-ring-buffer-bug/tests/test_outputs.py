import torch
import pytest
import sys
import os

# Add path to import the module
sys.path.append("/app/task-deps")
sys.path.append("/app")

# Determine import based on file existence (Agent might save to solution.py)
if os.path.exists("/app/solution.py"):
    from solution import KVCache
else:
    from kv_cache import KVCache

def test_cache_no_wrap():
    """
    Test standard behavior before the buffer fills up.
    This should pass even with the bug.
    """
    cache = KVCache(max_seq_len=5, hidden_dim=2)
    
    # Add 3 tokens: A, B, C
    t1 = torch.tensor([[1.0, 1.0]])
    t2 = torch.tensor([[2.0, 2.0]])
    t3 = torch.tensor([[3.0, 3.0]])
    
    cache.add(t1, t1)
    cache.add(t2, t2)
    cache.add(t3, t3)
    
    k_view, v_view = cache.get_ordered_view()
    
    # Expect [1, 2, 3]
    expected = torch.cat([t1, t2, t3], dim=0)
    
    assert torch.allclose(k_view, expected), "Basic cache filling failed."
    print("\nTest passed: No-wrap scenario works.")

def test_cache_wrap_around_ordering():
    """
    Test behavior when buffer wraps.
    Buffer Size: 4.
    Insert: 1, 2, 3, 4. State: [1, 2, 3, 4]. Pos: 0 (wrapped).
    Insert: 5.          State: [5, 2, 3, 4]. Pos: 1.
    
    Correct View Needed: [2, 3, 4, 5] (Oldest to Newest)
    Buggy View Returns:  [5, 2, 3, 4] (Physical layout)
    """
    cache = KVCache(max_seq_len=4, hidden_dim=1)
    
    # Fill cache
    for i in range(1, 5): # 1, 2, 3, 4
        val = torch.tensor([[float(i)]])
        cache.add(val, val)
        
    # Overflow
    val_5 = torch.tensor([[5.0]])
    cache.add(val_5, val_5)
    
    # Now verify order
    k_view, v_view = cache.get_ordered_view()
    
    # Expected order: 2, 3, 4, 5
    expected = torch.tensor([[2.0], [3.0], [4.0], [5.0]])
    
    print(f"\nCache Physical State (Dump): {cache.k_buffer.flatten().tolist()}")
    print(f"Returned View: {k_view.flatten().tolist()}")
    print(f"Expected View: {expected.flatten().tolist()}")
    
    if not torch.allclose(k_view, expected):
        pytest.fail("The KVCache returned tokens in the wrong order after wrapping! "
                    "Ensure you rotate the buffer so the oldest token is first.")
        
    print("Test passed: Wrap-around ordering is correct.")

def test_cache_wrap_multiple_times():
    """
    Stress test with multiple wraps.
    """
    cache = KVCache(max_seq_len=3, hidden_dim=1)
    # Add 1..10
    for i in range(1, 11):
        val = torch.tensor([[float(i)]])
        cache.add(val, val)
        
    # Should contain last 3: [8, 9, 10]
    expected = torch.tensor([[8.0], [9.0], [10.0]])
    k_view, _ = cache.get_ordered_view()
    
    if not torch.allclose(k_view, expected):
        pytest.fail(f"Deep wrap failed. Got {k_view.flatten().tolist()}, expected [8.0, 9.0, 10.0]")

    print("Test passed: Deep wrap works.")