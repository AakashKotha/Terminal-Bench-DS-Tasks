import json

def calculate_metrics(requests_file, batches_file, max_wait_sla=0.05):
    """
    Simulates the processing of requests based on the batches outputted.
    """
    # 1. Load Requests
    requests = {}
    with open(requests_file, 'r') as f:
        for line in f:
            r = json.loads(line)
            requests[r['request_id']] = r
            
    # 2. Load Batches
    batches = []
    with open(batches_file, 'r') as f:
        for line in f:
            batches.append(json.loads(line))
            
    # 3. Simulate Time
    # We assume the scheduler processes the input file sequentially.
    # The "Trigger Time" for a batch is determined by the arrival time of the LAST item in that batch.
    # Because you can't ship a batch until you have all the items.
    
    latencies = []
    batch_sizes = []
    processed_ids = set()
    
    for batch in batches:
        if not batch: continue
        
        batch_sizes.append(len(batch))
        
        # When did this batch become ready?
        # It's the max arrival time of items in it.
        # (Assuming the code ran linearly)
        try:
            arrival_times = [requests[rid]['arrival_time'] for rid in batch]
        except KeyError:
            return {"error": "Invalid Request ID in output"}
            
        batch_ready_time = max(arrival_times)
        
        # Calculate Queue Latency for each item
        # Latency = (Batch Ready Time) - (Item Arrival Time)
        # (We ignore GPU processing time for this specific 'Queue' metric, 
        # or assume it's constant/pipelined)
        
        for rid, arrival in zip(batch, arrival_times):
            latency = batch_ready_time - arrival
            latencies.append(latency)
            processed_ids.add(rid)
            
    # Metrics
    if not latencies:
        return {"error": "No requests processed"}
        
    avg_batch_size = sum(batch_sizes) / len(batch_sizes)
    
    # SLA Violation Rate
    violations = sum(1 for l in latencies if l > max_wait_sla)
    violation_rate = violations / len(latencies)
    
    # Coverage
    coverage = len(processed_ids) / len(requests)
    
    return {
        "avg_batch_size": avg_batch_size,
        "sla_violation_rate": violation_rate,
        "coverage": coverage,
        "max_latency": max(latencies)
    }