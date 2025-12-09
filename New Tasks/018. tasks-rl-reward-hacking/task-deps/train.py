import numpy as np
import os
import sys

# Import env
sys.path.append("/app/src")
from grid_env import WarehouseEnv

OUTPUT_MODEL = "/app/models/q_table.npy"

def train():
    env = WarehouseEnv()
    n_states = env.grid_size * env.grid_size
    n_actions = 4
    
    # Q-Table
    Q = np.zeros((n_states, n_actions))
    
    # Params
    lr = 0.1
    gamma = 0.9
    epsilon = 0.1
    n_episodes = 500
    
    success_count = 0
    recent_success = []
    
    print("Starting Training...")
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Epsilon Greedy
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, n_actions)
            else:
                action = np.argmax(Q[state, :])
                
            next_state, reward, done, _ = env.step(action)
            
            # Update
            old_value = Q[state, action]
            next_max = np.max(Q[next_state, :])
            new_value = (1 - lr) * old_value + lr * (reward + gamma * next_max)
            Q[state, action] = new_value
            
            state = next_state
            total_reward += reward
            
            # Check success (Reached Goal (3,3) -> Index 18)
            if done and reward >= 100: # Assuming goal gives large reward
                success_count += 1
                recent_success.append(1)
            elif done:
                recent_success.append(0)
                
        # Trim history
        if len(recent_success) > 100:
            recent_success.pop(0)
            
        if episode % 100 == 0:
            rate = sum(recent_success) / len(recent_success) if recent_success else 0.0
            print(f"Episode {episode}: Reward={total_reward}, Success Rate={rate:.2f}")

    final_rate = sum(recent_success) / len(recent_success)
    print(f"Training Complete. Final Success Rate: {final_rate:.2f}")
    
    # Save Model
    os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
    np.save(OUTPUT_MODEL, Q)
    
    if final_rate < 0.9:
        print("FAILURE: Agent failed to consistently solve the task.")
    else:
        print("SUCCESS: Agent mastered the task.")

if __name__ == "__main__":
    train()