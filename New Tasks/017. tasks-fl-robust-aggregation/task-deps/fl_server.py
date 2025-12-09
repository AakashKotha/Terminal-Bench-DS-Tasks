import numpy as np
import os
import json
from sklearn.metrics import accuracy_score

DATA_PATH = "/app/data/client_partitions.npz"
TEST_PATH = "/app/data/global_test.npz"
MODEL_PATH = "/app/models/global_model.npz"

class LogisticRegressionClient:
    def __init__(self, X, y, learning_rate=0.1):
        self.X = X
        self.y = y
        self.lr = learning_rate
        # Weights initialized later
        
    def train_step(self, global_weights):
        # Simulate local training (1 epoch of SGD)
        # Simple Logistic Regression Gradient: X.T * (sigmoid(Xw) - y)
        w = global_weights.copy()
        
        # Sigmoid
        logits = np.dot(self.X, w)
        preds = 1 / (1 + np.exp(-logits))
        
        # Gradient
        grad = np.dot(self.X.T, (preds - self.y)) / len(self.y)
        
        # Update
        w -= self.lr * grad
        return w

class MaliciousClient:
    """
    The Attacker.
    Returns weights that drift aggressively away from the true solution.
    """
    def __init__(self, X, y):
        pass # Doesn't care about data
        
    def train_step(self, global_weights):
        # POISON ATTACK:
        # Return weights that are massive and inverted
        # This drags the Mean arithmetic heavily.
        return -100.0 * global_weights + np.random.normal(0, 10, global_weights.shape)

def load_data():
    partitions = np.load(DATA_PATH, allow_pickle=True)
    test_data = np.load(TEST_PATH)
    
    clients = []
    # Clients 0-8 are Honest
    for i in range(9):
        c_data = partitions[f"client_{i}"].item()
        clients.append(LogisticRegressionClient(c_data["X"], c_data["y"]))
        
    # Client 9 is Malicious
    # Note: The agent might see this code and know client 9 is bad, 
    # but they must fix the aggregation logic, not just 'del clients[9]'.
    c_data = partitions["client_9"].item()
    clients.append(MaliciousClient(c_data["X"], c_data["y"]))
    
    return clients, test_data["X"], test_data["y"]

def aggregate_updates(client_weights):
    """
    INPUT: List of weight arrays from all clients.
    OUTPUT: Single aggregated weight array.
    """
    stack = np.stack(client_weights)
    
    # --- VULNERABLE CODE START ---
    # Standard FedAvg uses Mean.
    # Because Client 9 sends -100x weights, the Mean is destroyed.
    
    aggr = np.mean(stack, axis=0)
    
    # --- VULNERABLE CODE END ---
    
    return aggr

def evaluate(weights, X_test, y_test):
    logits = np.dot(X_test, weights)
    preds = (1 / (1 + np.exp(-logits))) > 0.5
    return accuracy_score(y_test, preds)

def run_federated_training():
    clients, X_test, y_test = load_data()
    
    # Init Global Model (20 features)
    global_weights = np.random.randn(20) * 0.01
    
    print("Starting Federated Learning (10 Clients, 1 Malicious)...")
    
    for round in range(20):
        local_updates = []
        
        # 1. Broadcast global weights & Train locally
        for client in clients:
            w_local = client.train_step(global_weights)
            local_updates.append(w_local)
            
        # 2. Secure Aggregation
        global_weights = aggregate_updates(local_updates)
        
        # 3. Evaluate
        acc = evaluate(global_weights, X_test, y_test)
        print(f"Round {round+1}: Validation Accuracy = {acc:.2%}")
        
    # Final Check
    if acc < 0.5:
        print("FAILURE: Model destroyed by poisoning.")
    else:
        print("SUCCESS: Model survived.")
        
    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    np.savez(MODEL_PATH, weights=global_weights, accuracy=acc)

if __name__ == "__main__":
    run_federated_training()