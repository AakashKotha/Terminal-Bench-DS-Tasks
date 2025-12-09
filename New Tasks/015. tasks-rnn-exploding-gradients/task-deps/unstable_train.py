import torch
import torch.nn as nn
import torch.optim as optim
import os

DATA_PATH = "/app/data/dataset.pt"
MODEL_PATH = "/app/models/stable_rnn.pt"

# --- MODEL DEFINITION ---
class DeepRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepRNN, self).__init__()
        # 4 Layers of Vanilla RNN is notoriously unstable
        # 'nonlinearity=relu' is unbounded, leading to explosion faster than tanh
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=4, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        # Predict for every time step
        out = self.fc(out)
        return out

def train():
    # Load Data
    if not os.path.exists(DATA_PATH):
        print("Data not found!")
        return
        
    data = torch.load(DATA_PATH)
    X = data['X']
    Y = data['Y']
    
    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Init Model
    model = DeepRNN(input_size=5, hidden_size=64, output_size=5)
    
    # --- HYPERPARAMETERS (FIXED BY MANAGEMENT) ---
    # High LR with Deep RNN = Explosion Risk
    learning_rate = 0.01 
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    print(f"Starting training with LR={learning_rate}...")
    
    model.train()
    for epoch in range(1, 6): # 5 Epochs
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Check for immediate failure
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"CRITICAL ERROR at Epoch {epoch}, Batch {batch_idx}: Loss is {loss.item()}")
                print("Gradient Explosion Detected! Training Aborted.")
                # We save the broken model anyway to prove failure
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                torch.save(model.state_dict(), MODEL_PATH)
                return # Exit early on crash

            loss.backward()
            
            # --- MISSING GRADIENT CLIPPING HERE ---
            # Without clip_grad_norm_, the gradients accumulate to infinity
            # --------------------------------------
            
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch} Loss: {epoch_loss / len(dataloader):.4f}")

    print("Training Complete Successfully.")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    train()