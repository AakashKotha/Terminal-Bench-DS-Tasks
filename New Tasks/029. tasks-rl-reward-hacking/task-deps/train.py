import numpy as np
import os
import sys
from grid_env import GridWorld

# Hyperparameters
ALPHA = 0.1    # Learning rate
GAMMA = 0.99   # Discount factor
EPSILON = 0.1  # Exploration rate
EPISODES = 2000
OUTPUT_FILE = "/app/output/q_table.npy"

def calculate_reward(current_pos, goal_pos, is_trap):
    """
    Calculates the reward for the current step.
    THIS IS THE SOURCE OF THE BUG.
    """
    if current_pos == goal_pos:
        return 10.0  # Big reward for reaching goal
    elif is_trap:
        return -10.0 # Penalty for hitting trap
    else:
        # --- BROKEN LOGIC START ---
        # The engineer thought: "Give a small reward to encourage movement!"
        # Actual effect: The agent learns to loop infinitely to accumulate this +0.1
        # until the max_steps timeout, gaining ~5.0 reward total, 
        # instead of reaching the goal quickly.
        return 0.1  
        # --- BROKEN LOGIC END ---

def train():
    env = GridWorld()
    # Q-Table: States x Actions (25 states, 4 actions)
    q_table = np.zeros((env.size * env.size, 4))
    
    print("Starting Training...")
    
    for episode in range(EPISODES):
        state_idx = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Epsilon-Greedy Action Selection
            if np.random.uniform(0, 1) < EPSILON:
                action = np.random.randint(0, 4) # Explore
            else:
                action = np.argmax(q_table[state_idx]) # Exploit
            
            # Take step
            next_state_idx, next_pos, done = env.step(action)
            
            # Calculate Reward
            is_trap = (next_pos == env.trap)
            reward = calculate_reward(next_pos, env.goal, is_trap)
            
            # Q-Learning Update
            # Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]
            old_value = q_table[state_idx, action]
            next_max = np.max(q_table[next_state_idx])
            
            new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
            q_table[state_idx, action] = new_value
            
            state_idx = next_state_idx
            total_reward += reward
            
        if episode % 200 == 0:
            print(f"Episode {episode}: Total Reward {total_reward:.2f}")

    print("Training Complete.")
    
    # Save Artifact
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    np.save(OUTPUT_FILE, q_table)
    print(f"Q-Table saved to {OUTPUT_FILE}")

    # Self-Verification Run
    print("\n--- Running Verification Episode ---")
    state_idx = env.reset()
    steps = 0
    path = []
    while steps < 20:
        action = np.argmax(q_table[state_idx])
        state_idx, pos, done = env.step(action)
        path.append(pos)
        steps += 1
        if done and pos == env.goal:
            print("SUCCESS: Reached Goal!")
            return
        if done:
            break
            
    print(f"FAILURE: Did not reach goal in 20 steps. Path: {path}")
    print("Diagnosis hint: Check the step reward.")

if __name__ == "__main__":
    train()