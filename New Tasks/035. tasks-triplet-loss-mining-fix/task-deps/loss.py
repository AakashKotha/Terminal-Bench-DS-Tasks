import torch

def get_distance_matrix(embeddings):
    """
    Computes pairwise euclidean distance.
    embeddings: (Batch, Emb_Dim)
    Returns: (Batch, Batch) matrix where entry (i,j) is distance between embedding i and j.
    """
    # (x-y)^2 = x^2 + y^2 - 2xy
    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = torch.diag(dot_product)
    
    # Broadcast subtraction
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    
    # Clamp to avoid negative values due to numerical errors
    distances = torch.clamp(distances, min=0.0)
    
    # We return squared euclidean distance usually, but for standard definition let's take sqrt
    # Adding epsilon for stability
    mask = (distances == 0.0).float()
    distances = distances + mask * 1e-16
    distances = torch.sqrt(distances)
    distances = distances * (1.0 - mask)
    
    return distances

def batch_hard_triplet_loss(embeddings, labels, margin=1.0):
    """
    Compute the Batch Hard Triplet Loss.
    
    For each anchor, we select:
    - The hardest positive (same class, max distance)
    - The hardest negative (diff class, min distance)
    
    Loss = Mean( max(d(a, p) - d(a, n) + margin, 0) )
    """
    dist_matrix = get_distance_matrix(embeddings)
    
    # Create masks
    # labels: (Batch,)
    # mask_pos: (Batch, Batch) -> 1 if i and j have same label
    mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    # mask_neg: (Batch, Batch) -> 1 if i and j have diff label
    mask_neg = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
    
    # --- 1. Get Hardest Positive (Max distance) ---
    # We multiply distance by mask. Unrelated pairs become 0.
    # Max will pick the furthest positive.
    # (This part is correct)
    positive_dists = dist_matrix * mask_pos
    hardest_positive_dist, _ = positive_dists.max(dim=1)
    
    # --- 2. Get Hardest Negative (Min distance) ---
    # We want the negative with the SMALLEST distance.
    
    # --- THE BUG IS HERE ---
    # Current broken logic:
    # It masks the positive pairs with 0. 
    # Then it takes the MAX of the remaining values.
    # This selects the negative that is FURTHEST away (easiest negative).
    # Since d(a, n) is very large, d(a, p) - d(a, n) + margin is likely < 0.
    # Result: Loss is 0.
    
    negative_dists = dist_matrix * mask_neg
    hardest_negative_dist, _ = negative_dists.max(dim=1) # <--- BUG: Using max() finds easiest negative!
    
    # -----------------------
    
    # Combine
    triplet_loss = torch.clamp(hardest_positive_dist - hardest_negative_dist + margin, min=0.0)
    
    return triplet_loss.mean()