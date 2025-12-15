import torch

def get_random_batch(batch_size=16, input_dim=128, num_classes=4):
    """
    Generates random embeddings and labels.
    """
    torch.manual_seed(42)
    embeddings = torch.randn(batch_size, input_dim)
    # Ensure normalized inputs effectively (though model does it too)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    labels = torch.randint(0, num_classes, (batch_size,))
    return embeddings, labels