# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# import shutil, os
# from detector import detect_image

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# @app.post("/detect/image")
# async def detect_single_image(file: UploadFile = File(...)):
#     path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     return detect_image(path)


# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse, JSONResponse
# import shutil, os

# from detector import detect_image
# from video_detector import detect_video

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# UPLOAD_DIR = "uploads"
# VIDEO_DIR = os.path.join(UPLOAD_DIR, "videos")

# os.makedirs(VIDEO_DIR, exist_ok=True)


# @app.post("/detect/image")
# async def detect_single_image(file: UploadFile = File(...)):
#     path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     return detect_image(path)


# @app.post("/detect/video")
# async def detect_video_api(file: UploadFile = File(...)):
#     path = os.path.join(VIDEO_DIR, file.filename)

#     with open(path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     if os.path.getsize(path) == 0:
#         return JSONResponse(
#             status_code=400,
#             content={"error": "Uploaded video is empty"}
#         )

#     result = detect_video(path)

#     if result.get("error"):
#         return JSONResponse(status_code=400, content=result)

#     return result


# @app.get("/video/{filename}")
# def serve_video(filename: str):
#     video_path = os.path.join(VIDEO_DIR, filename)
#     if not os.path.exists(video_path):
#         return JSONResponse(status_code=404, content={"error": "Video not found"})

#     return FileResponse(video_path, media_type="video/mp4")


# app.py (snippet - add these or replace relevant parts)

# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# import uuid, cv2
# import numpy as np
# from collections import defaultdict
# from ultralytics import YOLO
# from pesticide_data import get_pesticide_recommendation

# # --------------------------------------------------
# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# model = YOLO("model/best.pt")

# CONF_TH = 0.35
# IOU_TH = 0.45
# MIN_BOX_AREA_RATIO = 0.0005

# sessions = {}

# # --------------------------------------------------
# def iou(a, b):
#     xA, yA = max(a[0], b[0]), max(a[1], b[1])
#     xB, yB = min(a[2], b[2]), min(a[3], b[3])
#     inter = max(0, xB - xA) * max(0, yB - yA)
#     areaA = (a[2] - a[0]) * (a[3] - a[1])
#     areaB = (b[2] - b[0]) * (b[3] - b[1])
#     union = areaA + areaB - inter
#     return inter / union if union else 0

# # --------------------------------------------------
# class LiveSession:
#     def __init__(self):
#         self.leaves = {}   # leaf_id -> data
#         self.next_id = 1

#     def match_leaf(self, box):
#         for lid, leaf in self.leaves.items():
#             if iou(box, leaf["box"]) > IOU_TH:
#                 return lid
#         lid = self.next_id
#         self.next_id += 1
#         self.leaves[lid] = {
#             "box": box,
#             "frames": 0,
#             "conf_sum": 0,
#             "disease": None
#         }
#         return lid

#     def update(self, detections):
#         for d in detections:
#             lid = self.match_leaf(d["box"])
#             leaf = self.leaves[lid]
#             leaf["box"] = d["box"]
#             leaf["frames"] += 1
#             leaf["conf_sum"] += d["conf"]
#             leaf["disease"] = d["class"]

#     def summarize(self):
#         disease_map = defaultdict(list)

#         for leaf in self.leaves.values():
#             if leaf["frames"] == 0:
#                 continue
#             avg_conf = leaf["conf_sum"] / leaf["frames"]
#             if avg_conf >= CONF_TH:
#                 disease_map[leaf["disease"]].append(avg_conf)

#         if not disease_map:
#             return {
#                 "primary_disease": None,
#                 "severity": "NONE",
#                 "infection_ratio": 0,
#                 "avg_confidence": 0,
#                 "infected_leaves": 0,
#                 "total_leaves": len(self.leaves),
#                 "recommendation": {
#                     "type": "healthy",
#                     "message": "No disease detected."
#                 }
#             }

#         primary = max(disease_map, key=lambda k: len(disease_map[k]))
#         infected = len(disease_map[primary])
#         total = len(self.leaves)
#         ratio = infected / total
#         avg_conf = sum(disease_map[primary]) / infected

#         severity = "HIGH" if ratio > 0.3 else "MEDIUM" if ratio > 0.1 else "LOW"

#         return {
#             "primary_disease": primary,
#             "severity": severity,
#             "infection_ratio": round(ratio * 100, 2),
#             "avg_confidence": round(avg_conf * 100, 2),
#             "infected_leaves": infected,
#             "total_leaves": total,
#             "recommendation": get_pesticide_recommendation(primary, severity)
#         }

# # --------------------------------------------------
# @app.post("/live/start")
# def start_live():
#     sid = str(uuid.uuid4())
#     sessions[sid] = LiveSession()
#     return {"session_id": sid}

# # --------------------------------------------------
# @app.post("/live/frame")
# async def live_frame(frame: UploadFile = File(...), session_id: str = Form(...)):
#     if session_id not in sessions:
#         return JSONResponse(status_code=400, content={"error": "Invalid session"})

#     img = cv2.imdecode(
#         np.frombuffer(await frame.read(), np.uint8),
#         cv2.IMREAD_COLOR
#     )
#     h, w = img.shape[:2]
#     res = model(img, conf=0.25, verbose=False)[0]

#     detections, boxes_out = [], []

#     if res.boxes:
#         for b in res.boxes:
#             x1, y1, x2, y2 = map(float, b.xyxy[0])
#             conf = float(b.conf)
#             cls = model.names[int(b.cls)]
#             if ((x2-x1)*(y2-y1))/(w*h) < MIN_BOX_AREA_RATIO:
#                 continue

#             detections.append({
#                 "box": (x1,y1,x2,y2),
#                 "class": cls,
#                 "conf": conf
#             })

#             boxes_out.append({
#                 "x1":x1,"y1":y1,"x2":x2,"y2":y2,
#                 "class":cls,"conf":conf
#             })

#     sessions[session_id].update(detections)
#     return {"boxes": boxes_out}

# # --------------------------------------------------
# @app.post("/live/stop/{session_id}")
# def stop_live(session_id: str):
#     session = sessions.pop(session_id, None)
#     if not session:
#         return JSONResponse(status_code=404, content={"error":"Session not found"})
#     return session.summarize()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers.image_router import router as image_router
from routers.video_router import router as video_router
from routers.live_webcam_router import router as live_router
from routers.drone_router import router as drone_router

app = FastAPI(title="AI Plant Leaf Disease Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(image_router)
app.include_router(video_router)
app.include_router(live_router)
app.include_router(drone_router)
