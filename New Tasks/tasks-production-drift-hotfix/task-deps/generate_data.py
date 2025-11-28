import json
import random
import os
import numpy as np
from datetime import datetime, timedelta

OUTPUT_DIR = "/app/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_logs():
    print("Generating production logs with Schema Drift...")
    np.random.seed(42)
    random.seed(42)
    
    n_samples = 5000
    start_time = datetime(2024, 2, 15, 6, 0, 0)
    
    logs = []
    
    for i in range(n_samples):
        # Time progresses
        timestamp = start_time + timedelta(seconds=i*2)
        is_drifted = i > (n_samples // 2) # Drift starts halfway
        
        # Ground Truth Features
        is_mobile = random.choice([True, False])
        os_ver = random.choice(["14.0", "15.2", "17.1"])
        
        # 1. Generate User Agent (The problem source)
        if not is_drifted:
            # LEGACY FORMAT: "AppName/2.0 (iPhone; CPU OS 15_2 like Mac OS X)"
            if is_mobile:
                ua = f"ShopApp/2.0 (iPhone; CPU OS {os_ver.replace('.','_')} like Mac OS X)"
            else:
                ua = "ShopApp/2.0 (Macintosh; Intel Mac OS X 10_15_7)"
        else:
            # NEW FORMAT (Drift): "MobilePlatform (OS=iOS 15.2; Device=iPhone)"
            # Completely different structure!
            if is_mobile:
                ua = f"MobilePlatform (OS=iOS {os_ver}; Device=iPhone)"
            else:
                ua = "DesktopPlatform (OS=macOS; Device=Mac)"
        
        # 2. Generate Label (Target)
        # Mobile users convert more often (Signal)
        # If the agent fails to parse is_mobile, they lose this signal -> AUC drops.
        base_prob = 0.3 if is_mobile else 0.1
        converted = 1 if random.random() < base_prob else 0
        
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "user_agent": ua,
            "session_id": f"sess_{i}",
            "converted": converted
        }
        logs.append(log_entry)
        
    with open(f"{OUTPUT_DIR}/production_logs.jsonl", "w") as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")
            
    print(f"Generated {n_samples} logs. Drift started at index {n_samples//2}.")

if __name__ == "__main__":
    generate_logs()