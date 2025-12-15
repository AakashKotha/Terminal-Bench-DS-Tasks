import subprocess
import sys
import pytest
import os
import signal
import time

def test_distributed_training_runs_without_deadlock():
    """
    Executes the user's solution.py.
    Passes if the script completes successfully (exit code 0) within 30 seconds.
    Fails if the script hangs (timeout) or crashes.
    """
    solution_path = "/app/solution.py"
    
    if not os.path.exists(solution_path):
        pytest.fail("solution.py not found. Please save your fixed script to /app/solution.py")

    print(f"Executing {solution_path}...")

    # We run the script as a subprocess
    # The original buggy script hangs indefinitely.
    # The fixed script should finish 5 epochs * 10 steps * ~0.05s ~= 2.5 seconds + overhead.
    # We give a generous 30s timeout.
    
    start_time = time.time()
    
    try:
        proc = subprocess.run(
            [sys.executable, solution_path],
            capture_output=True,
            text=True,
            timeout=30  # Hard timeout to detect deadlock
        )
    except subprocess.TimeoutExpired:
        pytest.fail("TEST FAILED: The training script timed out! It is likely still deadlocked. Did you ensure all ranks participate in all_reduce?")

    duration = time.time() - start_time
    
    # 1. Check Exit Code
    if proc.returncode != 0:
        print("STDOUT:\n", proc.stdout)
        print("STDERR:\n", proc.stderr)
        pytest.fail(f"Script crashed with exit code {proc.returncode}")

    # 2. Check for completion
    # We expect 4 processes to print "Training Complete"
    completion_count = proc.stdout.count("Training Complete")
    if completion_count != 4:
         pytest.fail(f"Expected 4 workers to finish, but found {completion_count} completion messages.\nOutput Snippet:\n{proc.stdout[-500:]}")

    # 3. Code Inspection (Anti-Cheat)
    # Ensure the user didn't just delete the all_reduce or the loop
    with open(solution_path, "r") as f:
        content = f.read()
        
    if "dist.all_reduce" not in content:
        pytest.fail("The solution must still perform gradient synchronization (dist.all_reduce).")
        
    if "range(num_epochs)" not in content and "range(5)" not in content:
        pytest.fail("Do not modify the number of epochs (5) to bypass the test.")

    print(f"Test Passed! Simulation finished in {duration:.2f} seconds.")