import numpy as np
import scipy.io.wavfile as wav
import scipy.signal
import os
import glob
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

TRAIN_DIR = "/app/data/train"
TEST_DIR = "/app/data/iot_device"
MODEL_PATH = "/app/models/glass_detector.pkl"

def extract_features(file_path):
    # Read WAV file
    # scipy.io.wavfile returns (sample_rate, data)
    sr, audio = wav.read(file_path)
    
    # Normalize to float
    audio = audio.astype(np.float32) / 32768.0
    
    # --- BROKEN LOGIC START ---
    # We compute the Power Spectral Density (PSD) / Spectrogram
    # We use a fixed window size.
    # PROBLEM: The frequency resolution depends on 'sr'.
    # f = k * sr / nperseg
    # Since 'sr' changes between Train (48k) and Test (8k), 
    # the features (bins) represent completely different frequencies.
    # The model expects "Glass" energy at bin X (High Freq in 48k).
    # In 8k audio, Bin X corresponds to Low Freq (Background).
    
    frequencies, times, spectrogram = scipy.signal.spectrogram(audio, fs=sr, nperseg=256)
    
    # Simple feature: Mean energy per frequency bin across time
    # Feature vector size = 129 (nperseg/2 + 1)
    # We ignore the 'frequencies' array which tells us the truth!
    spectral_profile = np.mean(spectrogram, axis=1)
    
    # --- BROKEN LOGIC END ---
    
    return spectral_profile

def load_dataset(directory):
    X = []
    y = []
    files = glob.glob(os.path.join(directory, "*.wav"))
    print(f"Loading {len(files)} files from {directory}...")
    
    for f in files:
        feat = extract_features(f)
        X.append(feat)
        
        # Label parsing
        if "glass" in os.path.basename(f):
            y.append(1)
        else:
            y.append(0)
            
    return np.array(X), np.array(y)

def main():
    # 1. Train
    print("Training on High-Quality Lab Data...")
    X_train, y_train = load_dataset(TRAIN_DIR)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    print(f"Training Accuracy: {train_acc:.2f}") # Should be ~1.0
    
    # 2. Test on IoT
    print("Testing on Low-Quality IoT Data...")
    X_test, y_test = load_dataset(TEST_DIR)
    
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"IoT Test Accuracy: {test_acc:.2f}")
    
    if test_acc < 0.6:
        print("FAILURE: Model failed to generalize to IoT device.")
        print("HINT: Check sampling rates and feature invariance.")
    else:
        print("SUCCESS: Model works on IoT device!")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
            
if __name__ == "__main__":
    main()