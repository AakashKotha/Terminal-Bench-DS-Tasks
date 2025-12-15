import asyncio
import pytest
import subprocess
import sys
import time
import os
import socket

SERVER_PORT = 8888
SERVER_HOST = '127.0.0.1'

def send_request_sync(message):
    """Simple sync socket client for testing"""
    try:
        reader, writer = socket.socket(socket.AF_INET, socket.SOCK_STREAM), None
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((SERVER_HOST, SERVER_PORT))
        s.sendall(message.encode())
        data = s.recv(1024)
        s.close()
        return data.decode()
    except Exception as e:
        return str(e)

async def async_client(message):
    """Async wrapper"""
    reader, writer = await asyncio.open_connection(SERVER_HOST, SERVER_PORT)
    writer.write(message.encode())
    await writer.drain()
    data = await reader.read(100)
    writer.close()
    await writer.wait_closed()
    return data.decode()

@pytest.fixture(scope="module")
def server_process():
    # Determine which file to run: solution or original
    server_script = "/app/solution_server.py"
    if not os.path.exists(server_script):
        # Fallback for debugging the test itself, or if user overwrote original
        if os.path.exists("/app/task-deps/server.py"):
            server_script = "/app/task-deps/server.py"
        else:
            pytest.fail("Server script not found. Expected /app/solution_server.py")

    print(f"Starting server: {server_script}")
    
    # Start the server as a subprocess
    proc = subprocess.Popen(
        [sys.executable, server_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for startup
    time.sleep(2)
    
    yield proc
    
    # Teardown
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()

@pytest.mark.asyncio
async def test_concurrency_fix(server_process):
    """
    We confirm the fix by sending a Prediction request (which takes ~2s).
    Immediately after, we send a Health request.
    
    If the event loop is blocked, Health will wait 2s.
    If the event loop is free (offloaded), Health will return instantly.
    """
    
    # 1. Start the heavy prediction
    print("Sending PREDICT request...")
    predict_task = asyncio.create_task(async_client("POST /predict"))
    
    # Give it a tiny moment to hit the server and start blocking if buggy
    await asyncio.sleep(0.2)
    
    # 2. Send health check
    print("Sending HEALTH request...")
    start_time = time.time()
    health_response = await async_client("GET /health")
    duration = time.time() - start_time
    
    print(f"Health check took: {duration:.4f} seconds")
    print(f"Health response: {health_response}")
    
    # 3. Wait for prediction to finish cleaning up
    await predict_task
    
    # Assertions
    if duration > 1.0:
        pytest.fail(
            f"Health check took too long ({duration:.2f}s). \n"
            "This indicates the Event Loop was blocked by the prediction task. \n"
            "You must offload the model.predict call to a thread or process executor."
        )
        
    assert "OK" in health_response, "Health check did not return OK"
    print("Test Passed: Health check responded immediately during heavy load.")

def test_solution_file_exists():
    if not os.path.exists("/app/solution_server.py"):
        pytest.fail("Please save your solution to /app/solution_server.py")