import sys
import numpy as np

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}
 
    # Initialize base cases (t == 0)
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]
 
    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
 
        for y in states:
            # --- THE BUG IS HERE ---
            # Multiplying many small probabilities results in UNDERFLOW.
            # (prob, state) = max((V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states)
            # -----------------------
            
            (prob, state) = max((V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states)
            V[t][y] = prob
            newpath[y] = path[state] + [y]
 
        # Optional: Normalize to prevent underflow? 
        # Even with normalization, basic multiplication is risky without log-sum-exp logic
        # But here we don't normalize at all.
        path = newpath
 
    n = 0 - 1
    (prob, state) = max((V[n][y], y) for y in states)
    return (prob, path[state])

def run():
    # Load Data
    with open("/app/data/chromosome_segment.txt", "r") as f:
        sequence = f.read().strip()
    
    # Model Parameters (Known)
    states = ('I', 'E')
    start_p = {'I': 0.5, 'E': 0.5}
    trans_p = {
        'I': {'I': 0.99, 'E': 0.01},
        'E': {'I': 0.01, 'E': 0.99}
    }
    emit_p = {
        'I': {'A': 0.3, 'C': 0.2, 'G': 0.2, 'T': 0.3},
        'E': {'A': 0.1, 'C': 0.4, 'G': 0.4, 'T': 0.1}
    }
    
    print(f"Analyzing sequence length: {len(sequence)}")
    prob, result_path = viterbi(sequence, states, start_p, trans_p, emit_p)
    
    result_str = "".join(result_path)
    
    # Save
    import os
    os.makedirs("/app/output", exist_ok=True)
    with open("/app/output/predictions.txt", "w") as f:
        f.write(result_str)
        
    print("Done.")

if __name__ == "__main__":
    run()