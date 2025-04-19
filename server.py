import json
import uuid
from typing import List, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import uvicorn

import numpy as np
import cv2 as cv
import os
from data_preprocess import DataPreprocessor
from keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D as KDepthwiseConv2D
from trainer import Trainer

# specify path to model
model_dir = "model"
model_file = "keras_model.h5"
label_file = "labels.txt"

model_path = os.path.join(model_dir, model_file)
label_path = os.path.join(model_dir, label_file)
data_preprocessor = DataPreprocessor()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# load the model
class DepthwiseConv2DIgnoreGroups(KDepthwiseConv2D):
    def __init__(self, *args, groups=None, **kwargs):
        # discard the `groups` kwarg
        super().__init__(*args, **kwargs)

model = load_model(
  model_path,
  custom_objects={"DepthwiseConv2D": DepthwiseConv2DIgnoreGroups}
)

# Load the labels
class_names = open(label_path, "r").readlines()

trainer = Trainer(data_preprocessor, model, class_names, None)


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
#       "clientIdA": WebSocket,
#       "clientIdB": WebSocket,
#       ...
#   }
# }
rooms: Dict[str, Dict[str, WebSocket]] = {}

# test if the server is running
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/asl")
async def asl(
    image: UploadFile = File(...),          # the JPEG blob
    peerId: str      = Form(...)            # your peerId field
):
    '''
    Detect character from frame sent from client
    Params:
        -input : blob
        -output : json
    '''
    # Read raw bytes
    image_bytes = await image.read()              # image_bytes is a bytes object

    # Convert bytes → numpy array → OpenCV BGR image
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv.imdecode(nparr, cv.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse({"error": "could not decode image"}, status_code=400)
    
    detected, label, confidence = trainer.preprocess_and_predict(bgr)
    print(f"Detected: {detected}")
    print(f"Label: {label}")
    print(f"Confidence: {confidence}")
    confidence = float(confidence) if confidence is not None else 0.0
    
    return JSONResponse({
        "detected": detected,
        "label": label,
        "confidence": confidence,
        "peerId": peerId
    })
    

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
                    message = {
                        "type": msg_type,
                        "from": client_id,
                        **{k: v for k, v in data.items() if k not in ["type", "target"]}
                    }
                    print(f"Forwarding {msg_type} from {client_id} to {target_id}")
                    await rooms[room_id][target_id].send_text(json.dumps(message))
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
