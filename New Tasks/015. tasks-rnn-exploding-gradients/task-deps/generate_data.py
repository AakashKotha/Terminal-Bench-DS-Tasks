import torch
import os

OUTPUT_DIR = "/app/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_sequences():
    print("Generating Sequence Data...")
    torch.manual_seed(42)
    
    # Parameters
    N_SAMPLES = 1000
    SEQ_LEN = 40  # Long enough to cause BPTT explosion in vanilla RNN
    INPUT_DIM = 5
    
    # Generate random sine-wave like patterns
    # Shape: (N, Seq, Dim)
    X = torch.zeros(N_SAMPLES, SEQ_LEN, INPUT_DIM)
    Y = torch.zeros(N_SAMPLES, SEQ_LEN, INPUT_DIM)
    
    for i in range(N_SAMPLES):
        # Random frequency and phase
        freq = torch.rand(1) * 0.1
        phase = torch.rand(1) * 3.14
        t = torch.arange(SEQ_LEN).float()
        
        # Signal
        sig = torch.sin(t * freq + phase).unsqueeze(-1).repeat(1, INPUT_DIM)
        noise = torch.randn(SEQ_LEN, INPUT_DIM) * 0.05
        
        X[i] = sig + noise
        # Target: Next step prediction (shifted by 1, simplistic)
        # Actually let's just make target = signal * 1.5 + 0.5 to force weight growth
        Y[i] = (sig + noise) * 1.5 + 0.5

    # Save
    torch.save({"X": X, "Y": Y}, os.path.join(OUTPUT_DIR, "dataset.pt"))
    print(f"Saved {N_SAMPLES} sequences of length {SEQ_LEN}.")

if __name__ == "__main__":
    generate_sequences()