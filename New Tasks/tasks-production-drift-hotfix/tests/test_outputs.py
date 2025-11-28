import os
import pytest
import joblib
import pandas as pd
import sys

# Ensure we can import agent code
sys.path.append("/app/src")

MODEL_PATH = "/app/models/propensity_model_v2.joblib"

def test_model_artifact_exists():
    """Step 1: Did they run the training script?"""
    assert os.path.exists(MODEL_PATH), "Model file not found. Did you run /app/src/train.py?"

def test_extractor_logic_fixed():
    """
    Step 2: Implicit Diagnosis Check.
    We directly test their FeatureExtractor class on a known 'New Format' string.
    If it returns is_mobile=0, they didn't fix the regex.
    """
    try:
        from feature_extractor import FeatureExtractor
    except ImportError:
        pytest.fail("Could not import FeatureExtractor from /app/src/feature_extractor.py")
        
    extractor = FeatureExtractor()
    
    # Test Case: The NEW format that caused the drift
    drifted_log = [{
        "user_agent": "MobilePlatform (OS=iOS 17.1; Device=iPhone)",
        "converted": 0
    }]
    
    df = extractor.extract_features(drifted_log)
    
    # Check is_mobile
    assert df.iloc[0]["is_mobile"] == 1, \
        "Extractor failed to identify 'MobilePlatform...' as mobile. You must update the regex logic."
        
    # Check os_major (Optional but good)
    # Should extract 17 from "17.1"
    assert df.iloc[0]["os_major"] == 17, \
        f"Extractor failed to parse OS version. Expected 17, got {df.iloc[0]['os_major']}"

def test_model_performance():
    """
    Step 3: End-to-End Performance Check.
    Load their saved model and test it on synthetic data to ensure they actually trained it well.
    """
    model = joblib.load(MODEL_PATH)
    
    # Create a small synthetic test set representing the NEW reality
    X_test = pd.DataFrame([
        {"is_mobile": 1, "os_major": 17}, # High prob conversion
        {"is_mobile": 0, "os_major": 0},  # Low prob conversion
    ])
    
    probs = model.predict_proba(X_test)[:, 1]
    
    # Mobile should have higher probability
    assert probs[0] > probs[1], \
        "Model logic is inverted or weak. Mobile users should have higher propensity."
    
    # Mobile probability should be significant (>25%)
    assert probs[0] > 0.25, \
        f"Model predicts very low probability ({probs[0]:.2f}) for mobile users. Did training data contain correct labels?"