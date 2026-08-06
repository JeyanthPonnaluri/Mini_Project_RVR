import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json

app = FastAPI()

class EmulatedServer:
    def __init__(self):
        self.connections = []
        self.client_weights = {}
        self.total_clients = 3

    async def register(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)
        print(f"[SERVER] Client connected: {len(self.connections)}/{self.total_clients}")

    def unregister(self, websocket: WebSocket):
        if websocket in self.connections:
            self.connections.remove(websocket)

    async def broadcast_global(self, weights):
        print(f"[SERVER] Broadcasting consensus weights to {len(self.connections)} clients")
        for ws in self.connections:
            await ws.send_json({"type": "broadcast", "weights": weights})

server = EmulatedServer()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await server.register(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            if data["type"] == "weights_update":
                client_id = data["client_id"]
                weights = data["weights"]
                server.client_weights[client_id] = weights
                print(f"[SERVER] Received update from {client_id}")
                
                # Check if all clients checked in
                if len(server.client_weights) == server.total_clients:
                    # Run mock aggregation (average)
                    print("[SERVER] All clients checked in. Aggregating consensus...")
                    aggregated = [sum(x)/len(x) for x in zip(*server.client_weights.values())]
                    server.client_weights.clear()
                    await server.broadcast_global(aggregated)
                    
    except WebSocketDisconnect:
        server.unregister(websocket)
        print("[SERVER] Client disconnected")
