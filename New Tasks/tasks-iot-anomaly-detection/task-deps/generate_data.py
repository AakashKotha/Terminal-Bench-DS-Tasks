import json
import sqlite3
import pandas as pd
from datetime import datetime, timezone

# --- 1. Create Sensor Metadata ---
# This defines the "truth" for each sensor's threshold.
# Agents must read this file first.
metadata = {
    "s-01": {"location": "Basement", "threshold": 25.0},
    "s-02": {"location": "Attic", "threshold": 100.0},
    "s-03": {"location": "Garage", "threshold": 50.0}
}

with open("sensor_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Created sensor_metadata.json")

# --- 2. Create Sensor Readings Database ---
# We will create specific scenarios to "trap" naive agents.
# All timestamps are in UTC.
base_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

def to_ts(seconds_offset):
    """Converts a datetime to a Unix timestamp integer."""
    return int((base_time + pd.Timedelta(seconds=seconds_offset)).timestamp())

# Define the raw data
data = [
    # Sensor s-01 (Threshold: 25.0)
    {"timestamp": to_ts(0), "sensor_id": "s-01", "value": 10.5}, # OK
    {"timestamp": to_ts(1), "sensor_id": "s-01", "value": 12.0}, # OK
    {"timestamp": to_ts(2), "sensor_id": "s-01", "value": 26.0}, # ANOMALY START (1)
    {"timestamp": to_ts(3), "sensor_id": "s-01", "value": 30.2}, # ANOMALY (2)
    {"timestamp": to_ts(4), "sensor_id": "s-01", "value": 28.5}, # ANOMALY END (3) - This is a valid event
    {"timestamp": to_ts(5), "sensor_id": "s-01", "value": 15.0}, # Back to normal
    {"timestamp": to_ts(6), "sensor_id": "s-01", "value": 35.0}, # TRAP 1: Single spike, NOT an event
    {"timestamp": to_ts(7), "sensor_id": "s-01", "value": 10.0}, # Back to normal
    
    # Sensor s-02 (Threshold: 100.0)
    {"timestamp": to_ts(0), "sensor_id": "s-02", "value": 80.0}, # OK
    {"timestamp": to_ts(1), "sensor_id": "s-02", "value": 85.0}, # OK
    {"timestamp": to_ts(2), "sensor_id": "s-02", "value": 101.0},# TRAP 2: Consecutive count 1
    {"timestamp": to_ts(3), "sensor_id": "s-02", "value": 105.0},# TRAP 2: Consecutive count 2
    {"timestamp": to_ts(4), "sensor_id": "s-02", "value": 90.0}, # Reset. This is NOT a valid event (count < 3)
    {"timestamp": to_ts(5), "sensor_id": "s-02", "value": 88.0}, # OK
    
    # Sensor s-03 (Threshold: 50.0) - No anomalies
    {"timestamp": to_ts(0), "sensor_id": "s-03", "value": 40.0},
    {"timestamp": to_ts(1), "sensor_id": "s-03", "value": 42.0},
    {"timestamp": to_ts(2), "sensor_id": "s-03", "value": 45.0},
]

# Create DataFrame
df = pd.DataFrame(data)

# Create SQLite DB and table
con = sqlite3.connect("sensor_readings.db")
df.to_sql("readings", con, if_exists="replace", index=False)
con.close()

print("Created sensor_readings.db")