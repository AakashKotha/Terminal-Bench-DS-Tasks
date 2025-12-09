import json
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import sys

# Import the (hopefully fixed) extractor
sys.path.append("/app/src")
from feature_extractor import FeatureExtractor

DATA_PATH = "/app/data/production_logs.jsonl"
MODEL_PATH = "/app/models/propensity_model_v2.joblib"

def train():
    print("Loading logs...")
    with open(DATA_PATH, "r") as f:
        logs = [json.loads(line) for line in f]
    
    print("Extracting features...")
    extractor = FeatureExtractor()
    df = extractor.extract_features(logs)
    
    # Check if features are actually working
    mobile_rate = df["is_mobile"].mean()
    print(f"DEBUG: Detected Mobile Rate: {mobile_rate:.2%}")
    
    if mobile_rate < 0.2:
        print("WARNING: Mobile rate is suspiciously low. Did the extractor fail on the new data format?")
    
    X = df[["is_mobile", "os_major"]]
    y = df["converted"]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Eval
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"\nModel Trained. Test AUC: {auc:.4f}")
    
    if auc < 0.75:
        print("FAILURE: AUC is too low. Fix the feature extractor to recover signal!")
        # We don't exit(1) here to allow the test harness to catch the file, 
        # but the test harness will check the AUC.
    else:
        print("SUCCESS: AUC is healthy.")
        
    # Save
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()