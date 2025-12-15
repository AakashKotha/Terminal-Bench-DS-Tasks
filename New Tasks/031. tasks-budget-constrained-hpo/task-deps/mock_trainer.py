import argparse
import math
import sys
import os
import random

def calculate_score(optimizer, lr, layers):
    # Base noise
    score = random.uniform(0.0, 0.02)
    
    # 1. Categorical influence
    if optimizer == "sgd":
        # SGD is hard to tune in this fake scenario, maxes out lower
        base = 0.6
        # Penalty for wrong LR magnitude
        penalty = min(abs(math.log10(lr) - math.log10(0.1)), 2.0) * 0.1
        score += base - penalty
    elif optimizer == "rmsprop":
        base = 0.8
        penalty = min(abs(math.log10(lr) - math.log10(0.01)), 2.0) * 0.1
        score += base - penalty
    elif optimizer == "adam":
        base = 0.99
        # Optimal LR is 0.001
        lr_penalty = min(abs(math.log10(lr) - math.log10(0.001)), 3.0) * 0.2
        # Optimal Layers is 3
        layer_penalty = abs(layers - 3) * 0.05
        
        score += base - lr_penalty - layer_penalty
    else:
        return 0.0

    # Clip result
    return max(0.0, min(0.999, score))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, required=True, choices=["sgd", "adam", "rmsprop"])
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--layers", type=int, required=True)
    args = parser.parse_args()

    # Log the call to enforce budget in tests
    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "usage_log.txt")
    with open(log_path, "a") as f:
        f.write(f"optimizer={args.optimizer}, lr={args.learning_rate}, layers={args.layers}\n")

    # Calculate score
    score = calculate_score(args.optimizer, args.learning_rate, args.layers)
    
    # Output only the score to stdout so agent can parse it
    print(f"{score:.4f}")

if __name__ == "__main__":
    main()