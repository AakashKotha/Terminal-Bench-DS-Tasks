import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
import os
import random
import uuid
from datetime import datetime, timedelta

# Configuration
OUTPUT_RAW = "/build/output/raw"
OUTPUT_GOLDEN = "/build/output/golden"
os.makedirs(OUTPUT_RAW, exist_ok=True)
os.makedirs(OUTPUT_GOLDEN, exist_ok=True)

random.seed(42)
np.random.seed(42)

def generate_dataset():
    print("Generating dataset...")
    
    # 1. Generate Base Transactions
    accounts = [f"ACC_{i:04d}" for i in range(1, 21)]
    base_data = []
    
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    
    for _ in range(200):
        tx_id = str(uuid.uuid4())
        acc = random.choice(accounts)
        amount = round(random.uniform(-1000, 1000), 2)
        ts = start_time + timedelta(seconds=random.randint(1, 86400))
        base_data.append({
            "transaction_id": tx_id,
            "account_id": acc,
            "amount": amount,
            "timestamp": ts,
            "type": "TX",
            "ref_transaction_id": None
        })

    df = pd.DataFrame(base_data)
    
    # 2. Inject Complex Events (Traps)
    
    # Correction: Pick a random TX and create a CORRECTION event later
    target_tx = df.iloc[0]
    correction = {
        "transaction_id": str(uuid.uuid4()),
        "account_id": target_tx["account_id"],
        "amount": 500.00, # New corrected amount
        "timestamp": target_tx["timestamp"] + timedelta(minutes=5),
        "type": "CORRECTION",
        "ref_transaction_id": target_tx["transaction_id"]
    }
    df = pd.concat([df, pd.DataFrame([correction])], ignore_index=True)

    # Tombstone: Pick another TX and kill it
    target_tx_2 = df.iloc[1]
    tombstone = {
        "transaction_id": str(uuid.uuid4()),
        "account_id": target_tx_2["account_id"],
        "amount": 0.0,
        "timestamp": target_tx_2["timestamp"] + timedelta(minutes=10),
        "type": "TOMBSTONE",
        "ref_transaction_id": target_tx_2["transaction_id"]
    }
    df = pd.concat([df, pd.DataFrame([tombstone])], ignore_index=True)

    # Duplicates: Pick a TX and duplicate it with a slightly later timestamp (Simulating retry)
    # The later one should win. We will change the amount in the duplicate to verify LWW.
    target_tx_3 = df.iloc[2]
    duplicate = target_tx_3.copy()
    duplicate["timestamp"] = duplicate["timestamp"] + timedelta(seconds=1)
    duplicate["amount"] = 999.99 # This value should persist
    df = pd.concat([df, pd.DataFrame([duplicate])], ignore_index=True)

    # 3. Calculate Golden Answer (The Truth)
    # Sort by timestamp to handle LWW and events in order
    df_sorted = df.sort_values("timestamp")
    
    # Resolve Logic
    final_txs = {} # Map tx_id -> (amount, account_id)
    dropped_ids = set()
    
    # First pass: Resolve duplications by LWW (Last Write Wins) on transaction_id
    # We essentially replay the log. If we see a tx_id again, we overwrite if timestamp is newer.
    # However, our instructions say "Global Deduplication... based on timestamp". 
    # To simplify logical verification: we group by transaction_id and take the one with max timestamp.
    
    # Create a clean reliable dataframe first
    df_dedup = df_sorted.sort_values("timestamp").groupby("transaction_id", as_index=False).last()
    
    # Now apply business logic (Corrections and Tombstones)
    # We need to map ref_ids to actual updates.
    
    valid_amounts = {} # tx_id -> amount
    tx_to_account = {} # tx_id -> account_id
    
    # Load all TX type first
    for _, row in df_dedup.iterrows():
        if row["type"] == "TX":
            valid_amounts[row["transaction_id"]] = row["amount"]
            tx_to_account[row["transaction_id"]] = row["account_id"]
            
    # Apply Corrections and Tombstones
    for _, row in df_dedup.iterrows():
        if row["type"] == "TOMBSTONE":
            ref = row["ref_transaction_id"]
            if ref in valid_amounts:
                del valid_amounts[ref]
        elif row["type"] == "CORRECTION":
            ref = row["ref_transaction_id"]
            if ref in valid_amounts:
                valid_amounts[ref] = row["amount"]

    # Aggregate
    results = []
    for tx_id, amount in valid_amounts.items():
        results.append({
            "account_id": tx_to_account[tx_id],
            "amount": amount
        })
    
    result_df = pd.DataFrame(results)
    golden_df = result_df.groupby("account_id").agg(
        final_balance=("amount", "sum"),
        transaction_count=("amount", "count")
    ).reset_index()
    
    golden_df["final_balance"] = golden_df["final_balance"].round(2)
    golden_df = golden_df.sort_values("account_id").reset_index(drop=True)
    
    # Save Golden
    golden_df.to_parquet(f"{OUTPUT_GOLDEN}/golden_balances.parquet")
    print("Golden answer generated.")

    # 4. Split into Shards (The Challenge)
    shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(shuffled)
    s1, s2, s3 = shuffled.iloc[:n//3], shuffled.iloc[n//3:2*n//3], shuffled.iloc[2*n//3:]
    
    # Shard Alpha: CSV, String Amount, ISO Time
    s1_out = s1.copy()
    s1_out["amount"] = s1_out["amount"].apply(lambda x: f"${x:,.2f}")
    s1_out["timestamp"] = s1_out["timestamp"].apply(lambda x: x.isoformat())
    s1_out.to_csv(f"{OUTPUT_RAW}/shard_alpha.csv", index=False)
    
    # Shard Beta: JSONL, Float Amount, Epoch Time (ms)
    s2_out = s2.copy()
    s2_out["timestamp"] = s2_out["timestamp"].apply(lambda x: int(x.timestamp() * 1000))
    s2_out.to_json(f"{OUTPUT_RAW}/shard_beta.jsonl", orient="records", lines=True)
    
    # Shard Gamma: XML, Integer Cents, String Time
    root = ET.Element("transactions")
    for _, row in s3.iterrows():
        entry = ET.SubElement(root, "entry")
        ET.SubElement(entry, "transaction_id").text = str(row["transaction_id"])
        ET.SubElement(entry, "account_id").text = str(row["account_id"])
        ET.SubElement(entry, "type").text = str(row["type"])
        if row["ref_transaction_id"]:
            ET.SubElement(entry, "ref_transaction_id").text = str(row["ref_transaction_id"])
        
        # Convert to cents integer
        cents = int(row["amount"] * 100)
        ET.SubElement(entry, "amount_cents").text = str(cents)
        ET.SubElement(entry, "timestamp").text = row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

    tree = ET.ElementTree(root)
    tree.write(f"{OUTPUT_RAW}/shard_gamma.xml")
    print("Raw shards generated.")

if __name__ == "__main__":
    generate_dataset()