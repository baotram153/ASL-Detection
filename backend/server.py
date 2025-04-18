import json
import uuid
from typing import List, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data structure:
# rooms = {
#   "roomId123": {
#       "clients": {
#           "clientIdA": WebSocket,
#           "clientIdB": WebSocket,
#           ...
#       }
#   }
# }
rooms: Dict[str, Dict[str, WebSocket]] = {}

@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    """
    WebSocket endpoint for a given room_id.
    Each new WebSocket is assigned a unique client_id and added to the room.
    We'll broadcast "new-peer" events and handle direct signaling messages.
    """
    await websocket.accept()
    client_id = str(uuid.uuid4())  # generate a unique ID

    if room_id not in rooms:
        rooms[room_id] = {}

    # Each user in each room correspond to a websocket
    rooms[room_id][client_id] = websocket

    # Notify existing clients that a new client joined
    await broadcast_message(
        room_id,
        {
            "type": "new-peer",
            "clientId": client_id,
            "roomClients": list(rooms[room_id].keys())
        },
        exclude=client_id
    )

    print(f"Client {client_id} joined room {room_id}. Total: {len(rooms[room_id])}")

    try:
        while True:
            # Wait for a message from this client
            data_str = await websocket.receive_text()
            data = json.loads(data_str)

            # We expect messages containing at least a "type" field
            msg_type = data.get("type")
            
            # Inspect the message
            print(msg_type)

            # If it is a WebRTC signal, it might have a "target" ID, so we can forward the signal to that specific peer.
            if msg_type in ["offer", "answer", "ice-candidate"]:
                target_id = data.get("target")
                if target_id in rooms[room_id]:
                    # forward to the specific target
                    await rooms[room_id][target_id].send_text(json.dumps({
                        "type": msg_type,
                        "from": client_id,
                        **{k: v for k, v in data.items() if k not in ["type", "target"]}
                    }))
            else:
                # Handle other message types if needed
                pass

    except WebSocketDisconnect:
        # Client disconnected
        print(f"Client {client_id} disconnected from room {room_id}")
    finally:
        # Remove client from room
        del rooms[room_id][client_id]

        # Notify others that this client left
        await broadcast_message(
            room_id,
            {
                "type": "peer-left",
                "clientId": client_id
            },
            exclude=None
        )

        # If the room is now empty, remove it
        if len(rooms[room_id]) == 0:
            del rooms[room_id]


async def broadcast_message(room_id: str, message: dict, exclude: str = None):
    """
    Send a message to all WebSockets in the room, optionally excluding one client_id.
    """
    data_str = json.dumps(message)
    for cid, ws in rooms[room_id].items():
        if cid == exclude:
            continue
        await ws.send_text(data_str)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
