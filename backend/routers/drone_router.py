from fastapi import APIRouter, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
import cv2, asyncio, numpy as np, uuid
from ultralytics import YOLO
import state
from ai_live import LiveSession

router = APIRouter(prefix="/drone")
model = YOLO("model/best.pt")


async def push_status(session_id, payload):
    ws = state.WS_CLIENTS.get(session_id)
    if ws:
        await ws.send_json(payload)


# ===================== PAIR CREATE =====================
@router.post("/pair/create")
def create_pair():
    token = str(uuid.uuid4())[:8]
    session_id = str(uuid.uuid4())

    state.PAIR_TOKENS[token] = session_id
    state.DRONE_SESSIONS[session_id] = LiveSession()

    return {"pair_token": token, "session_id": session_id}


# ===================== PAIR ACCEPT =====================
@router.post("/pair/accept")
async def accept_pair(token: str = Form(...), device_id: str = Form(...)):
    if token not in state.PAIR_TOKENS:
        return JSONResponse(status_code=400, content={"error": "Invalid token"})

    session_id = state.PAIR_TOKENS.pop(token)

    await push_status(session_id, {
        "status": "connected",
        "device_id": device_id
    })

    return {"session_id": session_id}


# ===================== FRAME STREAM =====================
@router.post("/frame")
async def drone_frame(frame: UploadFile = File(...), session_id: str = Form(...)):

    session = state.DRONE_SESSIONS.get(session_id)

    # 🔥 STOP SIGNAL
    if not session:
        return JSONResponse(status_code=410, content={"error": "Stopped"})

    raw = await frame.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)

    res = model(img, conf=0.25, verbose=False)[0]

    detections = []
    if res.boxes:
        for b in res.boxes:
            detections.append({
                "class": model.names[int(b.cls)],
                "conf": float(b.conf)
            })

    session.update(detections)

    ok, jpeg = cv2.imencode(".jpg", img)
    if ok:
        state.LAST_FRAMES[session_id] = jpeg.tobytes()

    await push_status(session_id, {
        "status": "streaming",
        "detections": detections,
        "gps": state.GPS_DATA.get(session_id)
    })

    return {"ok": True}


# ===================== GPS =====================
@router.post("/gps")
async def drone_gps(
    session_id: str = Form(...),
    lat: float = Form(...),
    lng: float = Form(...)
):
    if session_id not in state.DRONE_SESSIONS:
        return JSONResponse(status_code=400, content={"error": "Invalid session"})

    state.GPS_DATA[session_id] = {"lat": lat, "lng": lng}

    await push_status(session_id, {
        "status": "gps",
        "gps": state.GPS_DATA[session_id]
    })

    return {"ok": True}


# ===================== STOP ANALYSIS =====================
@router.post("/stop-analysis/{session_id}")
async def stop_analysis(session_id: str):

    session = state.DRONE_SESSIONS.pop(session_id, None)

    if not session:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    # 🔥 THIS LINE WAS MISSING
    result = session.summarize()

    state.GPS_DATA.pop(session_id, None)
    state.LAST_FRAMES.pop(session_id, None)

    await push_status(session_id, {"status": "stopped"})

    return result



# ===================== MJPEG STREAM =====================
@router.get("/mjpeg/{session_id}")
async def mjpeg_stream(session_id: str):
    async def gen():
        while True:
            frame = state.LAST_FRAMES.get(session_id)
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    frame + b"\r\n"
                )
            await asyncio.sleep(0.05)

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ===================== WEBSOCKET =====================
@router.websocket("/ws/{session_id}")
async def ws_endpoint(ws: WebSocket, session_id: str):
    await ws.accept()
    state.WS_CLIENTS[session_id] = ws
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        state.WS_CLIENTS.pop(session_id, None)
