import json
import numpy as np
import os

OUTPUT_DIR = "/app/task_file/input_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_trace(filename, num_requests, rate_lambda):
    """
    Generates a Poisson arrival process.
    rate_lambda: requests per second.
    """
    data = []
    current_time = 0.0
    
    for i in range(num_requests):
        # Inter-arrival time (exponential distribution)
        dt = np.random.exponential(1.0 / rate_lambda)
        current_time += dt
        
        req = {
            "request_id": i,
            "arrival_time": current_time,
            "tokens": np.random.randint(10, 100)
        }
        data.append(req)
        
    with open(os.path.join(OUTPUT_DIR, filename), 'w') as f:
        for r in data:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    np.random.seed(42)
    # Bucket 1: Sparse traffic (5 reqs/sec). 
    # Waiting for 16 items takes ~3 seconds. SLA violation!
    generate_trace("requests_bucket_1.jsonl", 200, 5.0)
    
    # Bucket 2: Bursty traffic (50 reqs/sec).
    # Waiting for 16 items takes ~0.3s. Borderline.
    generate_trace("requests_bucket_2.jsonl", 1000, 50.0)
    
    print("Data generated.")