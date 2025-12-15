import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    """
    A simplified Graph Attention Layer.
    """
    def __init__(self, in_features, out_features):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight matrix for linear transformation
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)

        # Attention mechanism parameter
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, input_feats, adj_matrix):
        """
        Args:
            input_feats: Tensor of shape (N, in_features)
            adj_matrix: Binary adjacency matrix of shape (N, N). 1 indicates edge, 0 none.
        Returns:
            output_feats: Tensor of shape (N, out_features)
        """
        N = input_feats.size(0)
        
        # 1. Linear Transformation
        h = torch.mm(input_feats, self.W) # (N, out_features)

        # 2. Attention Mechanism
        # Prepare broadcasting for all pairs
        # h_repeated_in_chunks: (N * N, out_features) -> Repeats rows: node 0, node 0, ..., node 1, node 1...
        a_input = self._prepare_attentional_mechanism_input(h)
        
        # Calculate raw attention scores e_ij
        e = torch.matmul(a_input, self.a) # (N*N, 1)
        e = e.view(N, N) # (N, N) - e[i, j] is score between node i and j

        e = self.leakyrelu(e)

        # --- THE BUG IS HERE ---
        # The developer forgot to apply the graph structure (adj_matrix).
        # We perform Softmax over ALL nodes (fully connected attention).
        # We SHOULD mask out entries where adj_matrix == 0.
        
        attention = F.softmax(e, dim=1) 
        
        # -----------------------

        # 3. Aggregation
        h_prime = torch.matmul(attention, h) # (N, out_features)

        return F.elu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size(0)
        
        # Repeat for all pairs
        # Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        # Wh_repeated_alternating = Wh.repeat(N, 1)
        
        # Optimized implementation for broadcasting
        # This creates pairwise concatenations [Wh_i || Wh_j]
        # (N, 1, out)
        Wh_i = Wh.unsqueeze(1)
        # (1, N, out) 
        Wh_j = Wh.unsqueeze(0)
        
        # Broadcast add -> (N, N, 2*out) via cat
        # We do this manually to match the self.a dimensions (2*out, 1)
        # Actually easier to repeat explicitly for the matmul
        
        Wh_repeat_1 = Wh.repeat_interleave(N, dim=0)
        Wh_repeat_2 = Wh.repeat(N, 1)
        
        all_combinations_matrix = torch.cat([Wh_repeat_1, Wh_repeat_2], dim=1)
        
        return all_combinations_matrix