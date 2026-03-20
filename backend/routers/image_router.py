from fastapi import APIRouter, UploadFile, File
import uuid, os, shutil
from detector import detect_image

router = APIRouter(prefix="/detect", tags=["Image Detection"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/image")
async def detect_single_image(file: UploadFile = File(...)):
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")

    try:
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return detect_image(path)
    finally:
        if os.path.exists(path):
            os.remove(path)
