import numpy as np
import os

OUTPUT_DIR = "/app/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_dna():
    print("Generating DNA sequence...")
    np.random.seed(42)
    
    # Parameters
    # State 0: Intron (AT-rich), State 1: Exon (GC-rich)
    states = ["I", "E"]
    n_bases = 15000  # Long enough to cause underflow (Prob ~ 0.25^15000 is tiny)
    
    # Transition Matrix (Stay in state with high prob)
    #    I     E
    # I  0.99  0.01
    # E  0.01  0.99
    trans_mat = np.array([[0.99, 0.01], [0.01, 0.99]])
    
    # Emission Matrix
    #       A    C    G    T
    # I (0) 0.3  0.2  0.2  0.3  (High A/T)
    # E (1) 0.1  0.4  0.4  0.1  (High G/C)
    emissions = np.array([
        [0.3, 0.2, 0.2, 0.3],
        [0.1, 0.4, 0.4, 0.1]
    ])
    bases = ["A", "C", "G", "T"]
    
    # Generate
    sequence = []
    true_states = []
    
    current_state = 0 # Start in Intron
    
    for _ in range(n_bases):
        # Emit base
        base = np.random.choice(bases, p=emissions[current_state])
        sequence.append(base)
        true_states.append(states[current_state])
        
        # Transition
        current_state = np.random.choice([0, 1], p=trans_mat[current_state])
        
    # Save Data
    seq_str = "".join(sequence)
    state_str = "".join(true_states)
    
    with open(os.path.join(OUTPUT_DIR, "chromosome_segment.txt"), "w") as f:
        f.write(seq_str)
        
    # Save Ground Truth (Hidden from agent in task instructions, used by tests)
    # We hide it in a dotfile or separate dir so agent doesn't just copy it
    os.makedirs("/app/tests/hidden", exist_ok=True)
    with open("/app/tests/hidden/ground_truth.txt", "w") as f:
        f.write(state_str)
        
    print(f"Generated {n_bases}bp sequence.")

if __name__ == "__main__":
    generate_dna()