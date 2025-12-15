import pytest
import os
import subprocess
import json
# Import local helper
from tests.cost_model_for_tests import calculate_metrics

SCRIPT_PATH = "/app/task_file/scripts/baseline_packer.py"
DATA_DIR = "/app/task_file/input_data"
OUTPUT_DIR = "/app/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_agent_script(input_name):
    input_path = os.path.join(DATA_DIR, input_name)
    output_path = os.path.join(OUTPUT_DIR, f"batches_{input_name}")
    
    cmd = [
        "python3", SCRIPT_PATH,
        "--input", input_path,
        "--output", output_path
    ]
    subprocess.check_call(cmd)
    return input_path, output_path

def test_sparse_traffic_performance():
    """
    Test Bucket 1: Sparse traffic.
    Naive batching waits too long here.
    Agent must implement a timeout flush (e.g., if item waits > 50ms, send it).
    """
    input_path, output_path = run_agent_script("requests_bucket_1.jsonl")
    
    metrics = calculate_metrics(input_path, output_path, max_wait_sla=0.05)
    
    if "error" in metrics:
        pytest.fail(f"Metric calculation failed: {metrics['error']}")
        
    print(f"\n[Sparse Traffic] Metrics: {json.dumps(metrics, indent=2)}")
    
    assert metrics['coverage'] == 1.0, "Not all requests were processed."
    
    # Strict Latency Check
    # The baseline script will have ~100% violation rate here.
    # The fixed script should have < 5%.
    assert metrics['sla_violation_rate'] < 0.05, \
        f"Latency SLA Violated! {metrics['sla_violation_rate']:.1%} of requests waited > 50ms. " \
        "You likely need to implement a time-based trigger."

def test_efficiency_tradeoff():
    """
    Test Bucket 2: High traffic.
    Here, batching is easy.
    We verify the agent didn't just set MAX_BATCH_SIZE = 1 (Trivial solution).
    """
    input_path, output_path = run_agent_script("requests_bucket_2.jsonl")
    
    metrics = calculate_metrics(input_path, output_path, max_wait_sla=0.05)
    
    print(f"\n[Dense Traffic] Metrics: {json.dumps(metrics, indent=2)}")
    
    # Check Average Batch Size
    # If they just process 1 by 1, avg_batch_size will be 1.0.
    # We want efficient batching (>= 4.0) when traffic permits.
    assert metrics['avg_batch_size'] > 4.0, \
        f"Batching efficiency too low ({metrics['avg_batch_size']:.2f}). " \
        "Don't just set batch_size=1. Group items when they arrive close together."