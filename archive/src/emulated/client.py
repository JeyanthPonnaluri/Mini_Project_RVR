import asyncio
import websockets
import json
import sys

async def run_client(client_id, server_url):
    print(f"[{client_id}] Connecting to {server_url}...")
    try:
        async with websockets.connect(server_url) as websocket:
            print(f"[{client_id}] Connected. Preparing local weight update...")
            # Mock local training weight updates (e.g. random weights)
            weights = [0.12, -0.45, 0.78, 0.05, -0.22]
            
            # Send weights update
            payload = {
                "type": "weights_update",
                "client_id": client_id,
                "weights": weights
            }
            await websocket.send(json.dumps(payload))
            print(f"[{client_id}] Uploaded local weights: {weights}")
            
            # Wait for aggregated consensus broadcast
            response = await websocket.recv()
            data = json.loads(response)
            if data["type"] == "broadcast":
                consensus = data["weights"]
                print(f"[{client_id}] Received global aggregated consensus: {consensus}")
                print(f"[{client_id}] Optimization round sync complete!")
                
    except Exception as e:
        print(f"[{client_id}] Error: {str(e)}")

if __name__ == "__main__":
    client_id = sys.argv[1] if len(sys.argv) > 1 else "Hospital_Default"
    server_url = sys.argv[2] if len(sys.argv) > 2 else "ws://localhost:8000/ws"
    
    # Run client
    asyncio.run(run_client(client_id, server_url))
