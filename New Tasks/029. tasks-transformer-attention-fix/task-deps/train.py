import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from transformer import SimpleTransformer
import sys

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Task Config
VOCAB_SIZE = 20  # Numbers 0-19
SEQ_LEN = 10
DATASET_SIZE = 1000
BATCH_SIZE = 32
EPOCHS = 15       # Should converge quickly if fixed
D_MODEL = 64      # Dimension small enough to run on CPU fast, large enough to need scaling
N_HEADS = 4
N_LAYERS = 2
D_FF = 256
LR = 0.001

def generate_reverse_task_data(num_samples, seq_len, vocab_size):
    """Generates (input, target) pairs where target is the reverse of input."""
    X = torch.randint(1, vocab_size, (num_samples, seq_len)) # 0 is padding in theory, but we keep it simple
    Y = torch.flip(X, [1])
    return X, Y

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Generate Data
    X, Y = generate_reverse_task_data(DATASET_SIZE, SEQ_LEN, VOCAB_SIZE)
    X, Y = X.to(device), Y.to(device)

    # Model
    model = SimpleTransformer(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF, max_len=SEQ_LEN).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Starting training...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        # Simple full-batch shuffle loops or just random sampling for lightness
        indices = torch.randperm(DATASET_SIZE)
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]
        
        num_batches = DATASET_SIZE // BATCH_SIZE
        
        for i in range(num_batches):
            start = i * BATCH_SIZE
            end = start + BATCH_SIZE
            x_batch = X_shuffled[start:end]
            y_batch = Y_shuffled[start:end]
            
            optimizer.zero_grad()
            output = model(x_batch) # (Batch, Seq, Vocab)
            
            # Reshape for Loss
            loss = criterion(output.view(-1, VOCAB_SIZE), y_batch.view(-1))
            loss.backward()
            
            # Gradient clipping to help stability, though not the root cause here
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            preds = output.argmax(dim=-1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.numel()
            
        avg_loss = total_loss / num_batches
        accuracy = correct / total
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {accuracy:.4f}")
        
        # Save model checkpoint
        torch.save(model.state_dict(), "/app/output_model.pth")
        
        if accuracy > 0.95:
            print("Converged! Stopping early.")
            break

    # Final check
    print(f"Final Training Accuracy: {accuracy:.4f}")
    if accuracy < 0.5:
        print("FAILURE: Model failed to learn the task. Check the Attention mechanism.")
        sys.exit(1)
    else:
        print("SUCCESS: Model learned the task.")
        sys.exit(0)

if __name__ == "__main__":
    train()