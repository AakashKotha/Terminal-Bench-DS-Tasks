import asyncio
import json
import socket
from model import HeavyModel

# Global model instance
model = HeavyModel()

async def handle_client(reader, writer):
    data = await reader.read(1024)
    message = data.decode().strip()
    
    response = ""
    
    if message == "GET /health":
        # Health check should be instant
        response = "OK"
    
    elif message.startswith("POST /predict"):
        # --- THE BUG IS HERE ---
        # The developer marked the function async, but the model.predict call
        # is synchronous and CPU-bound.
        # Executing this line blocks the ENTIRE Event Loop.
        # No other coroutines (like pending health checks) can run until this finishes.
        
        result = model.predict("dummy input")
        
        # -----------------------
        response = json.dumps(result)
        
    else:
        response = "404 Not Found"

    writer.write(response.encode())
    await writer.drain()
    writer.close()

async def main():
    server = await asyncio.start_server(handle_client, '0.0.0.0', 8888)
    print("Serving on 8888...")
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass