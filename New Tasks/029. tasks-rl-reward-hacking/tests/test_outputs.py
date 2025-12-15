import numpy as np
import pytest
import os
import sys

# Append src to path to import env
sys.path.append("/app/src")
from grid_env import GridWorld

Q_TABLE_PATH = "/app/output/q_table.npy"

def test_q_table_exists():
    assert os.path.exists(Q_TABLE_PATH), "Q-Table artifact not found. Did the training script run?"

def test_agent_reaches_goal_efficiently():
    """
    Loads the agent's trained Q-table and simulates an episode.
    The Start is (0,0), Goal is (4,4).
    Optimal path length is 8 steps (e.g. 4 Right, 4 Down).
    We allow up to 15 steps to account for slight sub-optimality, 
    but infinite loops (the bug) will fail this.
    """
    q_table = np.load(Q_TABLE_PATH)
    env = GridWorld()
    state_idx = env.reset()
    
    steps = 0
    max_steps = 15
    reached_goal = False
    path = [(0,0)]
    
    for _ in range(max_steps):
        # Always exploit
        action = np.argmax(q_table[state_idx])
        
        state_idx, pos, done = env.step(action)
        path.append(pos)
        steps += 1
        
        if pos == env.goal:
            reached_goal = True
            break
        
        if done: # Battery died or trap
            break
            
    print(f"Test Agent Path: {path}")
    
    if not reached_goal:
        pytest.fail(f"Agent failed to reach goal in {max_steps} steps. The policy is likely stuck in a loop due to positive reward hacking.")
        
    assert len(path) <= 10, f"Agent took {len(path)} steps. Optimal is ~9. The reward function might be too lenient."