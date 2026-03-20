from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uuid, cv2, numpy as np
from ultralytics import YOLO

import state
from ai_live import LiveSession

router = APIRouter(prefix="/live", tags=["Live Webcam"])
model = YOLO("model/best.pt")

@router.post("/start")
def start_live():
    session_id = str(uuid.uuid4())
    state.DRONE_SESSIONS[session_id] = LiveSession()
    return {"session_id": session_id}

@router.post("/frame")
async def live_frame(frame: UploadFile = File(...), session_id: str = Form(...)):
    session = state.DRONE_SESSIONS.get(session_id)
    if not session:
        return JSONResponse(400, {"error": "Invalid session"})

    img = cv2.imdecode(
        np.frombuffer(await frame.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    res = model(img, conf=0.25, verbose=False)[0]
    boxes = [{
        "class": model.names[int(b.cls)],
        "conf": float(b.conf)
    } for b in res.boxes or []]

    session.update(boxes)
    return {"boxes": boxes}

@router.post("/stop/{session_id}")
def stop_live(session_id: str):
    session = state.DRONE_SESSIONS.pop(session_id, None)
    if not session:
        return JSONResponse(404, {"error": "Session not found"})
    return session.summarize()
