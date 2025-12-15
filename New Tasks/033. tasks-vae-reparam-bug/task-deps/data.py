import torch

def generate_synthetic_data(n_samples=1000, input_dim=20):
    """
    Generates data with a simple underlying latent structure.
    E.g., data = A * sin(x) + B
    """
    torch.manual_seed(42)
    # Create structured data: 
    # Half the data is close to 0, half close to 1, with patterns.
    data = torch.rand(n_samples, input_dim)
    
    # Make it learnable: Features 0-5 are correlated
    data[:, :5] = data[:, :5] > 0.5
    data = data.float()
    
    return data