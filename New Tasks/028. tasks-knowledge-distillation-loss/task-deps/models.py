import torch
import torch.nn as nn

class TeacherModel(nn.Module):
    def __init__(self, input_dim=20, num_classes=5):
        super().__init__()
        # Deep network, high capacity
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class StudentModel(nn.Module):
    def __init__(self, input_dim=20, num_classes=5):
        super().__init__()
        # Shallow network, low capacity
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.net(x)