import torch

class KVCache:
    def __init__(self, max_seq_len, hidden_dim):
        """
        A Key-Value Cache implemented as a Ring Buffer.
        """
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        
        # Buffer shape: (max_seq_len, hidden_dim)
        # We assume batch_size=1 for this simplified task
        self.k_buffer = torch.zeros(max_seq_len, hidden_dim)
        self.v_buffer = torch.zeros(max_seq_len, hidden_dim)
        
        self.size = 0       # Current number of elements stored
        self.write_pos = 0  # Index where the NEXT token will be written

    def add(self, k, v):
        """
        Adds a single token's key and value to the cache.
        k, v shape: (1, hidden_dim)
        """
        # Overwrite at current position
        self.k_buffer[self.write_pos] = k.squeeze(0)
        self.v_buffer[self.write_pos] = v.squeeze(0)
        
        # Advance pointer
        self.write_pos = (self.write_pos + 1) % self.max_seq_len
        
        # Update size (saturates at max_seq_len)
        self.size = min(self.size + 1, self.max_seq_len)

    def get_ordered_view(self):
        """
        Returns (Keys, Values) sorted chronologically.
        Shape: (current_seq_len, hidden_dim)
        """
        # --- THE BUG IS HERE ---
        # The developer simply returns the valid slice of the buffer.
        # This works if the buffer hasn't wrapped around yet.
        # IF we have wrapped (size == max_len), write_pos is somewhere in the middle.
        # Example: Buffer [C, D, E, A, B], write_pos is 3 (pointing to A).
        # Chronological order should be [A, B, C, D, E].
        # Current code returns [C, D, E, A, B].
        
        if self.size < self.max_seq_len:
            # Buffer not full, data is contiguous from 0 to size
            return self.k_buffer[:self.size], self.v_buffer[:self.size]
        else:
            # Buffer is full.
            # INCORRECT: Returning the raw buffer in physical order.
            # This breaks causality for the attention mechanism.
            return self.k_buffer, self.v_buffer