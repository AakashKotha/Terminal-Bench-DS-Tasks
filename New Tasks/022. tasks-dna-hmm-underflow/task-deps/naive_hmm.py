import numpy as np
import json
import os

DATA_PATH = "/app/data/genome_data.json"
OUTPUT_PATH = "/app/output/predictions.json"

class GeneHMM:
    def __init__(self):
        self.states = [0, 1] # 0=Intron, 1=Exon
        self.bases = ['A', 'C', 'G', 'T']
        
        # Probabilities (Linear Space)
        self.start_p = np.array([0.5, 0.5])
        
        self.trans_p = np.array([
            [0.99, 0.01], # 0 -> 0, 0 -> 1
            [0.01, 0.99]  # 1 -> 0, 1 -> 1
        ])
        
        # rows=states, cols=bases(A,C,G,T)
        self.emit_p = np.array([
            [0.4, 0.1, 0.1, 0.4], # State 0
            [0.1, 0.4, 0.4, 0.1]  # State 1
        ])
        
        self.base_map = {b: i for i, b in enumerate(self.bases)}

    def viterbi(self, sequence):
        """
        Naive Viterbi algorithm.
        V[t][s] = max probability of observing sequence up to t ending in state s.
        """
        T = len(sequence)
        N = len(self.states)
        
        # V[t, s] stores the probability
        V = np.zeros((T, N))
        
        # Backpointers to reconstruct path
        backptr = np.zeros((T, N), dtype=int)
        
        # 1. Initialization
        first_base_idx = self.base_map[sequence[0]]
        for s in range(N):
            # --- FLAW: Standard multiplication leads to underflow ---
            V[0, s] = self.start_p[s] * self.emit_p[s, first_base_idx]
            
        # 2. Recursion
        for t in range(1, T):
            base_idx = self.base_map[sequence[t]]
            for s in range(N):
                # Calculate prob of arriving at state 's' from all prev states 'prev'
                # max( V[t-1, prev] * transition[prev, s] * emission[s, obs] )
                
                probs = [V[t-1, prev] * self.trans_p[prev, s] * self.emit_p[s, base_idx] 
                         for prev in range(N)]
                
                max_p = np.max(probs)
                V[t, s] = max_p
                backptr[t, s] = np.argmax(probs)
            
            # Debug: Check for underflow early
            if np.sum(V[t, :]) == 0:
                print(f"CRITICAL WARNING: Underflow detected at step {t}. Probabilities collapsed to 0.0.")
                return None # Fail early

        # 3. Termination
        best_last_path_prob = np.max(V[T-1, :])
        best_last_state = np.argmax(V[T-1, :])
        
        # 4. Backtrack
        best_path = [best_last_state]
        for t in range(T-1, 0, -1):
            best_last_state = backptr[t, best_last_state]
            best_path.insert(0, best_last_state)
            
        return best_path

def main():
    print("Loading Data...")
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    
    seq = data['sequence']
    gt = data['ground_truth']
    
    print(f"Sequence Length: {len(seq)}")
    
    # Try running the HMM
    model = GeneHMM()
    
    # Test on a short substring first (sanity check)
    print("\nRunning Short Sequence Test (50bp)...")
    short_pred = model.viterbi(seq[:50])
    if short_pred:
        print("Short Test: SUCCESS")
    else:
        print("Short Test: FAILED")
        
    print("\nRunning Full Sequence Test (2000bp)...")
    long_pred = model.viterbi(seq)
    
    results = {}
    
    if long_pred is None:
        print("Long Test: FAILED (Underflow)")
        results["success"] = False
        results["accuracy"] = 0.0
    else:
        # Calculate Accuracy
        matches = sum([1 for i, j in zip(long_pred, gt) if i == j])
        acc = matches / len(gt)
        print(f"Long Test: SUCCESS. Accuracy: {acc:.2%}")
        results["success"] = True
        results["accuracy"] = acc
        results["predictions"] = long_pred
        
    # Save Output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()