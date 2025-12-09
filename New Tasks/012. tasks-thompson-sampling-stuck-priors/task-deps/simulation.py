import numpy as np
import json
import os
import sys

# Add src to path to import agent
sys.path.append("/app/src")
from bandit import ThompsonSampler

OUTPUT_PATH = "/app/output/performance.json"

def run_simulation():
    print("Starting Ad Campaign Simulation...")
    np.random.seed(42)
    
    # Ground Truth CTRs
    # Arm 0: 4%, Arm 1: 5%, Arm 2: 12% (WINNER), Arm 3: 4%, Arm 4: 5%
    true_ctrs = [0.04, 0.05, 0.12, 0.04, 0.05]
    n_arms = len(true_ctrs)
    optimal_arm = np.argmax(true_ctrs)
    
    n_steps = 5000
    agent = ThompsonSampler(n_arms)
    
    optimal_picks = 0
    total_reward = 0
    
    # Run Loop
    for t in range(n_steps):
        # 1. Agent picks arm
        chosen_arm = agent.select_arm()
        
        # 2. Metric tracking
        if chosen_arm == optimal_arm:
            optimal_picks += 1
            
        # 3. Environment feedback (Bernoulli Reward)
        # Simulate a click based on true CTR
        prob = true_ctrs[chosen_arm]
        reward = 1 if np.random.random() < prob else 0
        total_reward += reward
        
        # 4. Update Agent
        agent.update(chosen_arm, reward)
        
        if (t + 1) % 1000 == 0:
            print(f"Step {t+1}: Optimal Selection Rate so far: {optimal_picks / (t+1):.2%}")

    final_rate = optimal_picks / n_steps
    print(f"\nFinal Optimal Selection Rate: {final_rate:.2%}")
    
    # Check failure condition to give hints in logs
    if final_rate < 0.50:
        print("CRITICAL FAILURE: The agent is not converging. It's essentially guessing.")
        print("HINT: Check the magnitude of self.alphas and self.betas in __init__.")
    else:
        print("SUCCESS: The agent locked onto the best arm.")

    # Save results
    results = {
        "n_steps": n_steps,
        "optimal_selection_rate": final_rate,
        "total_clicks": total_reward
    }
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    run_simulation()