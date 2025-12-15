import time
import math

class HeavyModel:
    """
    Simulates a CPU-bound ML model (e.g., BERT on CPU).
    """
    def predict(self, input_text):
        # Simulate heavy CPU blocking work.
        # In a real scenario, this is a PyTorch/NumPy operation interacting with the CPU.
        # We calculate primes to burn CPU cycles for 2 seconds.
        
        # NOTE: We deliberately do NOT use time.sleep() here because time.sleep() 
        # releases the GIL in some implementations or is easily cheatable. 
        # We want to simulate actual CPU crunching that blocks the thread.
        
        target = 30000000
        count = 0
        # Simple busy wait to simulate blocking compute
        end_time = time.time() + 2.0
        while time.time() < end_time:
            count += 1
            if count % 1000 == 0:
                math.sqrt(count)
        
        return {"label": "positive", "confidence": 0.99}