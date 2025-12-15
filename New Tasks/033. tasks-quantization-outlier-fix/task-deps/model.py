import torch
import torch.nn as nn

class SimpleClassifier(nn.Module):
    """
    A simple MLP for classification.
    Input: 10 features
    Output: 2 classes
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x