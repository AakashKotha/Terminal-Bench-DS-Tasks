import re
import pandas as pd

class FeatureExtractor:
    def __init__(self):
        pass
    
    def extract_features(self, logs):
        """
        Input: List of dicts (raw logs)
        Output: DataFrame with features X and target y
        """
        data = []
        for log in logs:
            ua = log.get("user_agent", "")
            
            # --- BROKEN LOGIC STARTS HERE ---
            
            # Legacy Regex: Expects "iPhone; CPU OS 15_2 like Mac OS X"
            mobile_match = re.search(r"iPhone; CPU OS (\d+)_(\d+)", ua)
            
            if mobile_match:
                is_mobile = 1
                # Extract major version (e.g., 15)
                os_major = int(mobile_match.group(1))
            else:
                is_mobile = 0
                os_major = 0
            
            # --- BROKEN LOGIC ENDS HERE ---
            
            data.append({
                "is_mobile": is_mobile,
                "os_major": os_major,
                "converted": log["converted"]
            })
            
        return pd.DataFrame(data)