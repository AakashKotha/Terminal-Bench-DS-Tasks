import torch
import torch.nn as nn

class EmbeddingNet(nn.Module):
    """
    Simple embedding network mapping input dim -> output dim.
    """
    def __init__(self, input_dim=128, emb_dim=32):
        super(EmbeddingNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim)
        )

    def forward(self, x):
        output = self.fc(x)
        # Normalize embeddings to unit sphere is standard for triplet loss
        return torch.nn.functional.normalize(output, p=2, dim=1)