import numpy as np
import json
import os

OUTPUT_DIR = "/app/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_dna(length=1000):
    # States: 0=Intron (AT rich), 1=Exon (CG rich)
    states = []
    observations = []
    
    # Initial State
    current_state = 0 if np.random.rand() > 0.5 else 1
    
    # Transition Matrix
    #        Intron  Exon
    # Intron  0.99   0.01
    # Exon    0.01   0.99
    trans = [[0.99, 0.01], [0.01, 0.99]]
    
    # Emission Matrix
    #         A    C    G    T
    # Intron 0.4  0.1  0.1  0.4
    # Exon   0.1  0.4  0.4  0.1
    emit = [
        {'A': 0.4, 'C': 0.1, 'G': 0.1, 'T': 0.4}, # State 0
        {'A': 0.1, 'C': 0.4, 'G': 0.4, 'T': 0.1}  # State 1
    ]
    
    bases = ['A', 'C', 'G', 'T']
    
    for _ in range(length):
        states.append(current_state)
        
        # Emit
        probs = [emit[current_state][b] for b in bases]
        obs = np.random.choice(bases, p=probs)
        observations.append(obs)
        
        # Transition
        current_state = np.random.choice([0, 1], p=trans[current_state])
        
    return "".join(observations), states

if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate a Long Sequence that guarantees underflow in naive implementation
    seq, hidden = generate_dna(length=2000)
    
    data = {
        "sequence": seq,
        "ground_truth": hidden
    }
    
    with open(os.path.join(OUTPUT_DIR, "genome_data.json"), "w") as f:
        json.dump(data, f)
        
    print("Generated sequence length:", len(seq))