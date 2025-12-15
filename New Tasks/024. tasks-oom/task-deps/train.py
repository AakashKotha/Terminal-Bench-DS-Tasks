import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import psutil

# Simple process memory checker
def print_memory():
    process = psutil.Process(os.getpid())
    print(f"RAM Used: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def train_model():
    torch.manual_seed(42)
    
    # 1. Dummy Data
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    
    # 2. Simple Model
    model = nn.Linear(10, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("Starting training...")
    print_memory()
    
    # We use a list to store history for reporting
    loss_history = []
    
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        # --- THE MEMORY LEAK ---
        # Storing the tensor 'loss' keeps the entire computation graph 
        # (X, output, weights history) in memory for this step.
        loss_history.append(loss)
        # -----------------------
        
        if epoch % 20 == 0:
            # Calculate average loss so far
            current_avg = sum(loss_history) / len(loss_history)
            print(f"Epoch {epoch}: Loss {current_avg}")
            print_memory()
            
    print("Training finished.")

if __name__ == "__main__":
    train_model()