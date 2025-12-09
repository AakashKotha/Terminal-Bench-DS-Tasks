import os
import pickle
import numpy as np
import pytest
import scipy.io.wavfile as wav
import scipy.signal

MODEL_PATH = "/app/models/glass_detector.pkl"

def test_model_saved():
    assert os.path.exists(MODEL_PATH), "Model file not found. Did you fix the script and achieve high accuracy?"

def test_model_generalization():
    """
    We reproduce the test logic to confirm the agent didn't just hardcode 'print(Success)'.
    We generate a fresh synthetic 8kHz "Glass" sound and see if the model catches it.
    """
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        
    # 1. Generate Synthetic 8k Glass Sound (True Positive)
    sr = 8000
    t = np.linspace(0, 1.0, sr)
    # 3500 Hz tone (Glass signature)
    sig = 0.5 * np.sin(2 * np.pi * 3500 * t) + np.random.normal(0, 0.5, sr)
    
    # 2. Extract Features using the Agent's Logic?
    # We can't use the agent's logic because we don't know *how* they fixed it (did they resample to 8k? or 48k?).
    # So we must rely on the fact that if they fixed it, the model *expects* a specific input vector size.
    # BUT, to verify, we essentially need to replicate the pipeline.
    
    # Instead, we will trust the artifact generation from the main script execution in 'run-tests.sh' 
    # but we can check the *properties* of the model coefficients to see if they make sense.
    
    # Actually, a better check:
    # If the agent fixed it by resampling everything to, say, 16k.
    # Then the input vector size should be predictable (nperseg/2 + 1).
    # If they didn't fix it, the input size is likely still 129 (default nperseg=256).
    
    # We can't strictly assume the implementation details (what target SR they picked).
    # So we will rely on the *Self-Reported* success of the script (which runs inside the container)
    # AND a simple sanity check on the model object.
    
    assert hasattr(model, "coef_"), "Model is not a trained classifier."
    
    # We can try to infer if they fixed the physics. 
    # Ideally, we'd run their 'extract_features' function from the modified file.
    
    import sys
    sys.path.append("/app/src")
    try:
        from train_model import extract_features
    except ImportError:
        pytest.fail("Could not import extract_features from /app/src/train_model.py")
        
    # Create a temp file
    tmp_path = "/tmp/test_8k.wav"
    wav.write(tmp_path, sr, (sig*32767).astype(np.int16))
    
    # Extract
    feat = extract_features(tmp_path)
    
    # Predict
    pred = model.predict([feat])[0]
    
    assert pred == 1, \
        "Model failed to detect a clear 8kHz Glass signal. Did you implement resampling?"

def test_resampling_check():
    """
    Implicit Check: Inspect the source code for 'resample' keyword or logic.
    """
    with open("/app/src/train_model.py", "r") as f:
        code = f.read().lower()
        
    # They should use scipy.signal.resample or simple decimation
    has_resample = "resample" in code or "decimat" in code or "zoom" in code
    # Or they might manually handle indices? Unlikely.
    
    if not has_resample:
        # Maybe they just changed the nperseg dynamically based on SR?
        # That is also a valid valid solution! (nperseg = sr * window_duration)
        # So we check for 'sr' usage in spectrogram call.
        has_dynamic_sr = "fs=sr" in code or "fs=target_sr" in code
        # The broken code HAD fs=sr, but it didn't fix the bin alignment.
        # To fix bin alignment without resampling, you must interpolate the spectrogram.
        
        # We'll stick to the functional test (test_model_generalization) as the source of truth.
        pass