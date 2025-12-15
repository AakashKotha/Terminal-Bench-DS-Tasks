import sqlite3
import random
import os
import time
from datetime import datetime, timedelta

DB_DIR = "/app/db"
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "analytics.db")

def generate_db():
    print(f"Generating Database at {DB_PATH}...")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # 1. Create Tables (No Indices initially except Primary Key)
    cur.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT, country TEXT, signup_date TEXT)")
    cur.execute("CREATE TABLE logs (log_id INTEGER PRIMARY KEY, user_id INTEGER, event_type TEXT, timestamp TEXT)")
    
    conn.commit()
    
    # 2. Insert Users (10k)
    print("Inserting 10k Users...")
    countries = ["US", "UK", "CA", "DE", "FR", "JP", "IN", "BR"]
    users = []
    for i in range(10000):
        users.append((
            i, 
            f"user_{i}@example.com", 
            random.choice(countries),
            "2023-01-01"
        ))
    cur.executemany("INSERT INTO users VALUES (?,?,?,?)", users)
    conn.commit()
    
    # 3. Insert Logs (1M)
    # This simulates a heavy table scan if not indexed
    print("Inserting 1M Logs... (This might take 10s)")
    
    # Generate batch data
    batch_size = 50000
    total_logs = 1000000
    
    # Distribution: 
    # Some logs in Jan 2024 (Target), most elsewhere.
    start_ts = datetime(2023, 1, 1).timestamp()
    end_ts = datetime(2024, 12, 31).timestamp()
    
    for i in range(0, total_logs, batch_size):
        logs = []
        for j in range(batch_size):
            uid = random.randint(0, 9999)
            # 50% Login, 50% View
            evt = "LOGIN" if random.random() < 0.5 else "VIEW"
            
            # Random time
            ts = start_ts + random.random() * (end_ts - start_ts)
            dt_str = datetime.fromtimestamp(ts).isoformat(sep=' ')
            
            logs.append((None, uid, evt, dt_str))
            
        cur.executemany("INSERT INTO logs VALUES (?,?,?,?)", logs)
        print(f"Inserted {i + batch_size} logs...")
        
    conn.commit()
    
    # Verify count
    count = cur.execute("SELECT count(*) FROM logs").fetchone()[0]
    print(f"Database ready. Total logs: {count}")
    conn.close()

if __name__ == "__main__":
    generate_db()