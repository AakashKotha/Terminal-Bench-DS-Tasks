import json
import sys
import argparse

def run_scheduler(input_file, output_file):
    MAX_BATCH_SIZE = 16
    
    # Current Queue
    queue = []
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            req = json.loads(line)
            # req keys: 'request_id', 'arrival_time', 'tokens'
            
            queue.append(req)
            
            # --- THE BUGGY LOGIC ---
            # This scheduler ONLY flushes when the batch is full.
            # If traffic stops or is slow, items sit in 'queue' forever.
            # Also, it doesn't look at timestamps. It blindly simulates infinite speed processing 
            # effectively, but in a real discrete event simulator (the test runner),
            # waiting for the 16th item might take 5 seconds.
            
            if len(queue) >= MAX_BATCH_SIZE:
                batch_ids = [r['request_id'] for r in queue]
                fout.write(json.dumps(batch_ids) + "\n")
                queue = []
                
        # Flush remaining? 
        # The naive implementation often forgets this, or does it at the very end.
        # Even if we flush here, the latency damage is done for the items that waited.
        if queue:
            batch_ids = [r['request_id'] for r in queue]
            fout.write(json.dumps(batch_ids) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    run_scheduler(args.input, args.output)