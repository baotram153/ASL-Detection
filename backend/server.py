# server.py

import json
from typing import List, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# Allow CORS (adjust origins as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# A simple in-memory store of room_id -> list of active WebSockets
rooms: Dict[str, List[WebSocket]] = {}


@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    """
    A WebSocket endpoint for a given room_id.
    Clients connect to this endpoint to exchange
    signaling data for WebRTC.
    """
    await websocket.accept()

    if room_id not in rooms:
        rooms[room_id] = []
    rooms[room_id].append(websocket)

    print(f"WebSocket connected to room {room_id}. Total clients in room: {len(rooms[room_id])}.")

    try:
        while True:
            data = await websocket.receive_text()
            # Broadcast the received message to all clients in the same room except the sender
            for client_ws in rooms[room_id]:
                if client_ws != websocket:
                    await client_ws.send_text(data)

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    finally:
        # Remove from the room on disconnect
        rooms[room_id].remove(websocket)
        if not rooms[room_id]:
            del rooms[room_id]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
