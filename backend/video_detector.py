import cv2
import numpy as np
import os
from ultralytics import YOLO
from collections import defaultdict
from pesticide_data import get_pesticide_recommendation

model = YOLO("model/best.pt")
CONF = 0.3


def detect_video(video_path: str):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error": "Cannot open uploaded video"}

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return {"error": "Video has no readable frames"}

    h, w = first_frame.shape[:2]

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps) if fps and fps > 0 else 25

    base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(
        os.path.dirname(video_path),
        f"{base}_processed.mp4"
    )

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    if not writer.isOpened():
        cap.release()
        return {"error": "Failed to create output video"}

    disease_pixels = defaultdict(int)
    total_frames = 0

    writer.write(first_frame)
    total_frames += 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        result = model(frame, conf=CONF, verbose=False)[0]

        if result.masks is None:
            writer.write(frame)
            continue

        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for i, mask in enumerate(masks):
            cls = model.names[int(classes[i])].lower()
            if cls.startswith("healthy"):
                continue

            px = int(np.sum(mask > 0.5))
            disease_pixels[cls] += px

            mask_u8 = (mask * 255).astype("uint8")
            mask_u8 = cv2.resize(mask_u8, (w, h))
            frame[mask_u8 > 0] = (0, 0, 255)

        writer.write(frame)

    cap.release()
    writer.release()

    if not disease_pixels:
        return {
            "video": os.path.basename(out_path),
            "severity": "NONE",
            "primary_disease": None,
            "infected_leaf_percent": 0.0,
            "recommendation": {
                "type": "healthy",
                "message": "No disease detected in video"
            }
        }

    primary = max(disease_pixels, key=disease_pixels.get)
    percent = round(
        (disease_pixels[primary] / (w * h * total_frames)) * 100,
        2
    )

    severity = "HIGH" if percent > 40 else "MEDIUM" if percent > 15 else "LOW"
    recommendation = get_pesticide_recommendation(primary, severity)

    return {
        "video": os.path.basename(out_path),
        "severity": severity,
        "primary_disease": primary,
        "infected_leaf_percent": percent,
        "recommendation": recommendation
    }
