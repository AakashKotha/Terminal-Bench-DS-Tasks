import torch

def generate_clean_data(n_samples=1000):
    """
    Generates clean synthetic data for classification.
    X ~ Normal(0, 1)
    Y based on simple linear boundary.
    """
    torch.manual_seed(42)
    X = torch.randn(n_samples, 10)
    # Simple rule: if sum of first 3 features > 0, class 1, else 0
    logits = X[:, :3].sum(dim=1)
    y = (logits > 0).long()
    return X, y

def generate_calibration_data_with_outliers(n_samples=200):
    """
    Generates calibration data that contains massive outliers.
    """
    torch.manual_seed(101)
    X, y = generate_clean_data(n_samples)
    
    # Inject corrupt outliers
    # About 1% of data has massive values
    n_outliers = int(n_samples * 0.01)
    indices = torch.randperm(n_samples)[:n_outliers]
    
    # Inject values like 500.0, -1000.0
    # Normal data range is approx [-3, 3]
    X[indices] = X[indices] * 500.0 
    
    return X