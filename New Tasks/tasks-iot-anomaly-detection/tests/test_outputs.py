import os
import json
import pytest
import sqlite3
import pandas as pd
from datetime import datetime, timezone

REPORT_PATH = "/app/reports/anomaly_report.json"
DB_PATH = "/app/data/sensor_readings.db"
META_PATH = "/app/data/sensor_metadata.json"

def get_golden_answer():
    """
    This function re-implements the task logic correctly to find the "golden"
    answer. This is what the agent's output will be compared against.
    """
    
    # 1. Load metadata
    with open(META_PATH, 'r') as f:
        metadata = json.load(f)

    # 2. Load sensor data from DB
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM readings ORDER BY sensor_id, timestamp", con)
    con.close()

    # 3. Find events by iterating and maintaining state
    events = []
    current_event = None

    for sensor_id, group in df.groupby('sensor_id'):
        threshold = metadata[sensor_id]['threshold']
        consecutive_count = 0
        
        for _, row in group.iterrows():
            is_over_threshold = row['value'] > threshold
            
            if is_over_threshold:
                consecutive_count += 1
                
                if current_event is None:
                    # Start a new potential event
                    current_event = {
                        "sensor_id": row['sensor_id'],
                        "start_ts": row['timestamp'],
                        "end_ts": row['timestamp'],
                        "peak_value": row['value'],
                        "count": 1
                    }
                else:
                    # Continue the current event
                    current_event["end_ts"] = row['timestamp']
                    current_event["peak_value"] = max(current_event["peak_value"], row['value'])
                    current_event["count"] += 1
            
            else:
                # Value is not over threshold, reset
                if current_event is not None and current_event["count"] >= 3:
                    # This event is valid, save it
                    events.append(current_event)
                
                # Reset for next event
                current_event = None
                consecutive_count = 0
        
        # Check at the end of the group
        if current_event is not None and current_event["count"] >= 3:
            events.append(current_event)
        
        current_event = None # Reset for next sensor

    # 4. Format the final report
    final_report = []
    for event in events:
        final_report.append({
            "sensor_id": event["sensor_id"],
            "event_start_time": datetime.fromtimestamp(event["start_ts"], tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "event_end_time": datetime.fromtimestamp(event["end_ts"], tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "peak_value": event["peak_value"]
        })
        
    return final_report


def test_anomaly_report_exists():
    """Test 1: Check if the output JSON file was created."""
    assert os.path.exists(REPORT_PATH), f"Expected file not found at {REPORT_PATH}"

def test_anomaly_report_content():
    """
    Test 2: Check if the content of the JSON report is correct,
    specifically checking the "3 consecutive" logic.
    """
    
    # 1. Get the "golden answer" by re-running the logic
    expected_data = get_golden_answer()
    
    # 2. Load the agent's output file
    try:
        with open(REPORT_PATH, 'r') as f:
            agent_data = json.load(f)
    except json.JSONDecodeError:
        pytest.fail(f"Could not decode JSON from {REPORT_PATH}. File might be empty or malformed.")
    except IOError:
        pytest.fail(f"Could not open file {REPORT_PATH}.")

    # 3. Validate the content
    assert isinstance(agent_data, list), "Output JSON should be a list of event objects."
    
    # This is the key check. The test data includes:
    # - s-01: one valid event of 3 readings
    # - s-01: one single spike (should be ignored)
    # - s-02: one event of 2 readings (should be ignored)
    # The final report should ONLY have the 1 valid event.
    assert len(agent_data) == len(expected_data), \
        f"Expected {len(expected_data)} anomalous event(s), but found {len(agent_data)}. " \
        "Did you correctly filter for *3 or more consecutive* readings?"

    # Sort both lists by start time to ensure comparison is not order-dependent
    sorted_agent_data = sorted(agent_data, key=lambda x: x['event_start_time'])
    sorted_expected_data = sorted(expected_data, key=lambda x: x['event_start_time'])

    # 4. Compare the sorted lists
    for agent_event, expected_event in zip(sorted_agent_data, sorted_expected_data):
        assert agent_event["sensor_id"] == expected_event["sensor_id"], \
            f"Event sensor_id mismatch. Expected {expected_event['sensor_id']}, got {agent_event['sensor_id']}"
        
        assert agent_event["event_start_time"] == expected_event["event_start_time"], \
            f"Event event_start_time mismatch for {agent_event['sensor_id']}. Check timestamp conversion and logic."
        
        assert agent_event["event_end_time"] == expected_event["event_end_time"], \
            f"Event event_end_time mismatch for {agent_event['sensor_id']}. Check timestamp conversion and logic."

        assert agent_event["peak_value"] == pytest.approx(expected_event["peak_value"]), \
            f"Event peak_value mismatch for {agent_event['sensor_id']}. Expected {expected_event['peak_value']}, got {agent_event['peak_value']}"

    print(f"\nSuccessfully validated {REPORT_PATH}.")
    print(f"Found {len(agent_data)} valid anomalous event(s), which matches the expected count.")