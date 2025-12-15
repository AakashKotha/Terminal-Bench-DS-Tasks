import numpy as np
from nanograd import Tensor

np.random.seed(1337)

class Layer:
    def __init__(self, nin, nout):
        self.w = Tensor(np.random.uniform(-1, 1, (nin, nout)))
        self.b = Tensor(np.zeros((nout,)))  # Bias is (nout,)
    
    def __call__(self, x):
        # x is (Batch, nin), w is (nin, nout) -> (Batch, nout)
        # b is (nout,) -> broadcasts to (Batch, nout)
        return x.matmul(self.w) + self.b 
    
    def parameters(self):
        return [self.w, self.b]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x).relu()
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# Synthetic Data
# Function: y = 2*x0 - 3*x1 + 1
X_train = np.random.randn(50, 2)
y_train = (2 * X_train[:, 0] - 3 * X_train[:, 1] + 1).reshape(-1, 1)

model = MLP(2, [8, 1])

# Optimizer
lr = 0.01

print(f"Initial bias shape layer 0: {model.layers[0].b.data.shape}")

for k in range(50):
    # Forward pass
    inputs = Tensor(X_train)
    targets = Tensor(y_train)
    
    # MLP output
    scores = model(inputs)
    
    # MSE Loss
    diff = scores - targets
    loss = (diff * diff).sum()
    
    # Zero grad
    for p in model.parameters():
        p.grad = np.zeros_like(p.data)
    
    # Backward
    loss.backward()
    
    # Update
    for p in model.parameters():
        # Sanity check: If the bug exists, p.grad might have shape (Batch, nout)
        # while p.data has shape (nout,).
        # This assert catches the shape mutation before it propagates
        if p.grad.shape != p.data.shape:
            print(f"CRITICAL ERROR: Gradient shape mismatch! Param: {p.data.shape}, Grad: {p.grad.shape}")
            print("Hint: This usually happens when the backward pass doesn't handle broadcasting correctly.")
            exit(1)
            
        p.data -= lr * p.grad
    
    print(f"Step {k}, loss {loss.data}")

print("Training finished successfully.")