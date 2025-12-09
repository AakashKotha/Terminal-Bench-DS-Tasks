import numpy as np

class GradientSanitizer:
    def __init__(self, clip_norm=1.0, noise_multiplier=0.0):
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier

    def sanitize_gradients(self, batch_gradients):
        """
        Input: batch_gradients of shape (batch_size, num_params)
        Output: Aggregated (mean) gradient of shape (num_params,)
        """
        
        # --- VULNERABLE LOGIC START ---
        # The developer thought: "I need to prevent gradients from exploding."
        # So they averaged them first.
        
        aggregated_grad = np.mean(batch_gradients, axis=0)
        
        # Then they clipped the result.
        # This bounds the update size, but it DOES NOT bound the influence 
        # of a single row in 'batch_gradients' BEFORE the mean.
        
        total_norm = np.linalg.norm(aggregated_grad)
        scale = min(1.0, self.clip_norm / (total_norm + 1e-6))
        
        clipped_grad = aggregated_grad * scale
        
        # --- VULNERABLE LOGIC END ---
        
        # Add DP Noise (Simulated)
        if self.noise_multiplier > 0:
            noise_std = self.clip_norm * self.noise_multiplier
            noise = np.random.normal(0, noise_std, size=clipped_grad.shape)
            return clipped_grad + noise
            
        return clipped_grad