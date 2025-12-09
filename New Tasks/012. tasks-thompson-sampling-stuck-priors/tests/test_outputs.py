import os
import json
import pytest
import numpy as np
import sys

# Ensure we can import the agent class to check logic directly if needed
sys.path.append("/app/src")

RESULTS_PATH = "/app/output/performance.json"

def test_simulation_results_exist():
    assert os.path.exists(RESULTS_PATH), "Results file not found. Did you run the simulation?"

def test_convergence_performance():
    """
    Validates that the agent learned the optimal arm.
    """
    with open(RESULTS_PATH, 'r') as f:
        data = json.load(f)
        
    rate = data["optimal_selection_rate"]
    
    print(f"Verified Selection Rate: {rate:.2%}")
    
    # 0.20 is random guessing (1/5).
    # 0.90+ means it found the winner early and stuck to it.
    assert rate > 0.90, \
        f"FAILED: Optimal Selection Rate is too low ({rate:.2%}). " \
        "The agent failed to learn. Did you weaken the priors (alphas/betas) to allow exploration?"

def test_prior_initialization_check():
    """
    Implicit Logic Check:
    Instantiate the class and inspect the priors.
    They should NOT be 1000.0 anymore.
    """
    from bandit import ThompsonSampler
    agent = ThompsonSampler(n_arms=5)
    
    # Check if priors are small enough to allow learning
    # Uninformative prior (1,1) sum is 2.
    # Weak prior (1, 10) sum is 11.
    # The trap was 2000.
    
    prior_strength = agent.alphas[0] + agent.betas[0]
    
    assert prior_strength < 100.0, \
        f"FAILED: Priors are still too strong (Strength={prior_strength}). " \
        "High strength (alpha+beta) reduces variance, preventing the model from updating its beliefs based on new data. " \
        "Set alpha=1, beta=1 (Uniform) or similar."