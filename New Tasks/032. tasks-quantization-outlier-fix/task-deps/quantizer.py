import torch

class Quantizer:
    def __init__(self, model):
        self.model = model
        self.scale = 1.0
        self.zero_point = 0
        
        # Buffers to store stats during calibration
        self.observed_data = []
        
        # Initial stats
        self.min_val = float('inf')
        self.max_val = float('-inf')

    def update_stats(self, x):
        """
        Observes input data x to calculate min/max range.
        Currently implements strict MinMax observation.
        """
        # --- THE BUG IS HERE ---
        # The agent needs to modify this logic.
        # Current logic: purely max/min driven.
        # If x contains an outlier (e.g., 1000.0), max_val becomes 1000.0.
        # If normal data is [-3, 3], the resolution becomes approx 1000/255 = 4.
        # All normal data [-3, 3] gets crushed into 0 or 1.
        
        batch_min = x.min().item()
        batch_max = x.max().item()
        
        if batch_min < self.min_val:
            self.min_val = batch_min
        if batch_max > self.max_val:
            self.max_val = batch_max
            
        # Store data for percentile calculation if needed
        # (Hint: You might need to change how you store or process this)
        self.observed_data.append(x)

    def compute_params(self):
        """
        Calculates scale and zero_point based on observed stats.
        """
        # If the user fixes update_stats to use percentiles, 
        # self.min_val and self.max_val should reflect the clipped range.
        
        # Ensure we have a valid range
        if self.min_val == float('inf'):
            self.min_val = -1.0
            self.max_val = 1.0
            
        # Symmetric quantization typically (simplification for this task)
        # We assume 8-bit quantization [0, 255]
        
        data_range = self.max_val - self.min_val
        if data_range == 0:
            data_range = 1.0
            
        self.scale = data_range / 255.0
        self.zero_point = round(-self.min_val / self.scale)
        
        # Clamp zero_point to int8 range (unsigned)
        self.zero_point = max(0, min(255, self.zero_point))
        
        print(f"Calibration Result -> Min: {self.min_val:.2f}, Max: {self.max_val:.2f}")
        print(f"Computed Scale: {self.scale:.4f}, Zero Point: {self.zero_point}")

    def quantize_forward(self, x):
        """
        Simulates the quantization effect:
        Float -> Quantize -> Dequantize -> Float
        This shows the precision loss.
        """
        # 1. Quantize
        # q = round(x / S + Z)
        x_q = torch.round(x / self.scale + self.zero_point)
        x_q = torch.clamp(x_q, 0, 255)
        
        # 2. Dequantize
        # x_hat = (q - Z) * S
        x_hat = (x_q - self.zero_point) * self.scale
        
        return x_hat

    def calibrate(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                # In a real scenario, we'd hook activations.
                # Here, for simplicity, we treat the input 'batch' as the activation
                # we want to quantize to demonstrate the outlier effect on the input layer.
                self.update_stats(batch)
        self.compute_params()