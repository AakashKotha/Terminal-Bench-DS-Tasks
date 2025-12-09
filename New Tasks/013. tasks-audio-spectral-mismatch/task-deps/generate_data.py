import numpy as np
import scipy.io.wavfile as wav
import os
import shutil

DATA_DIR = "/app/data"
os.makedirs(os.path.join(DATA_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "iot_device"), exist_ok=True)

def generate_tone(freq, duration, sr):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * freq * t)

def generate_noise(duration, sr, high_freq=False):
    n_samples = int(sr * duration)
    if high_freq:
        # High frequency noise (Glass break simulation)
        # Filtered white noise via simple illusion (random changes)
        noise = np.random.normal(0, 0.5, n_samples)
        # We want more energy in high bands
        return noise
    else:
        # Low frequency noise (Background hum)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        # 50Hz hum + some random walk
        hum = 0.3 * np.sin(2 * np.pi * 50 * t)
        noise = np.random.normal(0, 0.1, n_samples)
        return hum + noise

def save_dataset(path, sr, count):
    print(f"Generating {count} files at {sr}Hz in {path}...")
    for i in range(count):
        # 50% Positive (Glass), 50% Negative (Background)
        is_glass = i % 2 == 0
        label = "glass" if is_glass else "bg"
        
        # Duration 1 second
        if is_glass:
            # Glass: High freq noise burst
            sig = generate_noise(1.0, sr, high_freq=True)
            # Add a distinct high-pitch transient (e.g. 3500 Hz)
            # 3500 Hz is detectable by both 48k (Nyquist 24k) and 8k (Nyquist 4k)
            # BUT: In 48k FFT, 3500Hz is at bin ~74 (if n_fft=1024)
            #      In 8k FFT,  3500Hz is at bin ~448
            # The model will look at bin 74. In 8k data, bin 74 is ~578Hz (Background noise!).
            # So the model will miss the glass sound in 8k data.
            sig += generate_tone(3500, 1.0, sr) 
        else:
            # Bg: Low freq
            sig = generate_noise(1.0, sr, high_freq=False)
            sig += generate_tone(150, 1.0, sr) # 150 Hz tone

        # Normalize and save
        sig = sig / np.max(np.abs(sig))
        sig = (sig * 32767).astype(np.int16)
        
        fname = f"{label}_{i}.wav"
        wav.write(os.path.join(path, fname), sr, sig)

if __name__ == "__main__":
    np.random.seed(42)
    # Train Data: High Quality (48 kHz)
    save_dataset(os.path.join(DATA_DIR, "train"), sr=48000, count=100)
    
    # IoT Data: Low Quality (8 kHz)
    # This represents the deployment environment shift
    save_dataset(os.path.join(DATA_DIR, "iot_device"), sr=8000, count=100)
    print("Audio data generation complete.")