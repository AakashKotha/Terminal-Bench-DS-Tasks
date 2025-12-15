import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Split heads: (batch, seq_len, d_model) -> (batch, seq_len, n_heads, head_dim)
        # Then transpose to: (batch, n_heads, seq_len, head_dim)
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # --- THE BUG IS HERE ---
        # Scaled Dot-Product Attention: softmax(QK^T / sqrt(d_k)) V
        # The implementer forgot the scaling factor math.sqrt(self.head_dim).
        # Without this, for large dimensions, dot products are large magnitude, 
        # pushing softmax into regions with extremely small gradients.
        
        energy = torch.matmul(Q, K.transpose(-2, -1))
        
        # If fixed, it should look like:
        # energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # Mask format assumed (batch, 1, 1, seq_len) or broadcastable
            energy = energy.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(energy, dim=-1)
        
        out = torch.matmul(attention, V)
        
        # Reshape back: (batch, n_heads, seq_len, head_dim) -> (batch, seq_len, n_heads, head_dim)
        # -> (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.fc_out(out)

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # ReLU activation
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        _x = x
        x = self.attention(x, x, x, mask)
        x = self.norm1(_x + self.dropout(x))
        _x = x
        x = self.ff(x)
        x = self.norm2(_x + self.dropout(x))
        return x

class SimpleTransformer(nn.Module):
    """
    A simplified Transformer for Sequence Modeling.
    Uses embeddings and multiple Transformer blocks.
    """
    def __init__(self, input_dim, d_model, n_heads, n_layers, d_ff, max_len=100, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.fc_out = nn.Linear(d_model, input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        x = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return self.fc_out(x)