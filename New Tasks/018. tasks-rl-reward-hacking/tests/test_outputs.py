import numpy as np
import os
import pytest
import sys

# We need the env logic (modified by agent) to run the test
sys.path.append("/app/src")
from grid_env import WarehouseEnv

MODEL_PATH = "/app/models/q_table.npy"

def test_model_artifact_exists():
    assert os.path.exists(MODEL_PATH), "Q-Table artifact not found. Did training run?"

def test_agent_performance():
    """
    Run the agent's trained Q-table greedily.
    It MUST reach the goal.
    If it gets stuck in a loop, it fails.
    """
    try:
        Q = np.load(MODEL_PATH)
    except Exception as e:
        pytest.fail(f"Failed to load Q-table: {e}")
        
    env = WarehouseEnv()
    state = env.reset()
    done = False
    steps = 0
    path = []
    
    # Run 1 episode strictly greedy
    while not done and steps < 20: # Should perform it quickly
        action = np.argmax(Q[state, :])
        state, _, done, _ = env.step(action)
        path.append(state)
        steps += 1
        
    # Check if goal reached
    # Goal is at (3,3) -> Index 3*5 + 3 = 18
    GOAL_IDX = 18
    
    assert state == GOAL_IDX, \
        f"Agent failed to reach goal. Final state: {state}. Steps: {steps}. Path: {path}.\n" \
        "Likely Cause: Agent is stuck in a reward loop or wandering."

def test_reward_function_fix():
    """
    Heuristic check: Did they add a step penalty or reduce the distraction reward?
    We verify the environment logic by instantiating it and checking step returns.
    """
    env = WarehouseEnv()
    env.reset()
    
    # Move to recharge station (1,1) -> 1*5+1 = 6
    # From (0,0) -> Down(1) -> (1,0) -> Right(3) -> (1,1)
    env.step(1)
    _, reward_at_station, _, _ = env.step(3)
    
    # The original buggy reward was +10.
    # To fix it, they usually reduce it, make it 0, or add a step penalty (-1).
    # If they kept it at +10 WITHOUT a step penalty, the loop persists.
    
    # If reward is still high (>1), we check if they added a step penalty elsewhere.
    # But generally, a robust fix reduces this "distractor" reward.
    
    # Strict check: Step reward should not be excessively positive for non-goal state
    assert reward_at_station < 5, \
        f"Reward at recharge station is still high ({reward_at_station}). " \
        "This encourages looping. Reduce it or add a negative step cost."