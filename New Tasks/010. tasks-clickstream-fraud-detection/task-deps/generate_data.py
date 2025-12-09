import json
import numpy as np
import os
import random

OUTPUT_DIR = "/app/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_sessions():
    print("Generating clickstream data...")
    np.random.seed(42)
    random.seed(42)
    
    n_humans = 500
    n_bots = 300
    
    # Define "Button" locations (Where humans usually click)
    # e.g., [Submit] at (960, 800), [Menu] at (100, 50)
    buttons = [
        (960, 800), # Center bottom CTA
        (1800, 50), # Top right profile
        (200, 500), # Sidebar link
    ]
    
    sessions = []
    
    # 1. Generate Humans
    for i in range(n_humans):
        sid = f"user_{i:05d}"
        n_clicks = random.randint(5, 20)
        
        events = []
        t = 1700000000.0
        
        for _ in range(n_clicks):
            # Human Timing: Irregular (Exponential/Poisson)
            # They pause to read, then click.
            dt = np.random.exponential(scale=3.0) + 0.5 # Min 0.5s reaction
            t += dt
            
            # Human Space: Gaussian around buttons (with some error)
            target = random.choice(buttons)
            x = int(np.random.normal(loc=target[0], scale=20))
            y = int(np.random.normal(loc=target[1], scale=20))
            
            # Clamp to screen
            x = max(0, min(1920, x))
            y = max(0, min(1080, y))
            
            events.append({"timestamp": t, "x": x, "y": y})
            
        sessions.append({"session_id": sid, "events": events, "label": "human"})

    # 2. Generate Bots
    for i in range(n_bots):
        sid = f"bot_{i:05d}"
        n_clicks = random.randint(5, 20) # Same volume as humans!
        
        events = []
        t = 1700000000.0
        
        # Strategy: The bot uses a fixed script.
        # "Sleep(2.0)"
        
        # Bot Timing: Extremely Low Variance (Fixed)
        base_delay = 2.0
        jitter = 0.01 # Tiny jitter to fool simple '==' checks
        
        for _ in range(n_clicks):
            dt = base_delay + random.uniform(-jitter, jitter)
            t += dt
            
            # Bot Space: Random Uniform (Simulating "Activity" to avoid idle timeout)
            # They just click anywhere on the screen
            x = random.randint(0, 1920)
            y = random.randint(0, 1080)
            
            events.append({"timestamp": t, "x": x, "y": y})
            
        sessions.append({"session_id": sid, "events": events, "label": "bot"})
        
    # Shuffle and Save
    random.shuffle(sessions)
    
    # Save the problem file (without labels)
    with open(os.path.join(OUTPUT_DIR, "session_logs.jsonl"), "w") as f:
        for s in sessions:
            record = {"session_id": s["session_id"], "events": s["events"]}
            f.write(json.dumps(record) + "\n")
            
    # Save the ground truth (hidden)
    os.makedirs("/app/tests/hidden", exist_ok=True)
    with open("/app/tests/hidden/labels.json", "w") as f:
        truth = {s["session_id"]: s["label"] for s in sessions}
        json.dump(truth, f)
        
    print(f"Generated {len(sessions)} sessions ({n_humans} humans, {n_bots} bots).")

if __name__ == "__main__":
    generate_sessions()