# ==============================
# IMPORTS
# ==============================
import cv2
import numpy as np
import random
from ultralytics import YOLO
from collections import defaultdict
from pesticide_data import get_pesticide_recommendation

# ==============================
# LOAD MODEL
# ==============================
model = YOLO("model/best.pt")
CONF_THRESHOLD = 0.30


# ==============================
# SEVERITY LOGIC
# ==============================
def calculate_severity(area_ratio, regions):
    if area_ratio > 0.40 or regions >= 40:
        return "HIGH"
    elif area_ratio > 0.15 or regions >= 15:
        return "MEDIUM"
    elif area_ratio > 0.01 or regions >= 3:
        return "LOW"
    return "NONE"


def severity_reason(area_percent, regions):
    return (
        f"Severity calculated using {area_percent}% infected leaf area "
        f"and {regions} infected regions."
    )


# ==============================
# VISUALIZATION (OPTIONAL)
# ==============================
def draw_disease_contours(image, disease_masks):
    out = image.copy()
    h, w = image.shape[:2]

    for mask in disease_masks:
        resized = cv2.resize(mask.astype("uint8"), (w, h))
        contours, _ = cv2.findContours(
            resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            if cv2.contourArea(cnt) < 40:
                continue

            cv2.drawContours(out, [cnt], -1, (0, 0, 255), 2)

            x, y, bw, bh = cv2.boundingRect(cnt)
            for _ in range(5):
                cx = random.randint(x, x + bw)
                cy = random.randint(y, y + bh)
                cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)

    return out


# ==============================
# MAIN DETECTION
# ==============================
def detect_image(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image")

    h, w = img.shape[:2]
    image_area = h * w

    result = model(img, conf=CONF_THRESHOLD, verbose=False)[0]

    # --------------------------
    # NO MASK = HEALTHY
    # --------------------------
    if result.masks is None:
        return {
            "detections": [],
            "infected_regions": 0,
            "infected_leaf_percent": 0.0,
            "severity": "NONE",
            "severity_reason": "",
            "primary_disease": None,
            "overlay_image": None,
            "all_diseases": [],
            "recommendation": {
                "type": "healthy",
                "message": "No disease detected. Crop looks healthy."
            }
        }

    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    disease_stats = defaultdict(lambda: {"pixels": 0, "regions": 0})
    disease_masks = []
    detections = []

    # --------------------------
    # PROCESS ONLY DISEASE MASKS
    # --------------------------
    for i, mask in enumerate(masks):
        if confidences[i] < CONF_THRESHOLD:
            continue

        class_name = model.names[int(classes[i])].lower()

        # ✅ VERY IMPORTANT FIX
        if class_name.startswith("healthy"):
            continue

        infected_pixels = int(np.sum(mask > 0.5))
        if infected_pixels == 0:
            continue

        disease_stats[class_name]["pixels"] += infected_pixels
        disease_stats[class_name]["regions"] += 1
        disease_masks.append(mask)

        detections.append({
            "disease": class_name,
            "confidence": round(float(confidences[i]) * 100, 2)
        })

    # --------------------------
    # NO DISEASE AFTER FILTER
    # --------------------------
    if not detections:
        return {
            "detections": [],
            "infected_regions": 0,
            "infected_leaf_percent": 0.0,
            "severity": "NONE",
            "severity_reason": "",
            "primary_disease": None,
            "overlay_image": None,
            "all_diseases": [],
            "recommendation": {
                "type": "healthy",
                "message": "No disease detected."
            }
        }

    # --------------------------
    # CALCULATIONS
    # --------------------------
    total_infected_pixels = sum(v["pixels"] for v in disease_stats.values())
    total_regions = sum(v["regions"] for v in disease_stats.values())

    area_ratio = total_infected_pixels / image_area
    infected_leaf_percent = round(min(area_ratio * 100, 100.0), 2)

    severity = calculate_severity(area_ratio, total_regions)
    reason = severity_reason(infected_leaf_percent, total_regions)

    primary_disease = max(
        disease_stats.items(), key=lambda x: x[1]["pixels"]
    )[0]

    recommendation = get_pesticide_recommendation(primary_disease, severity)

    overlay_img = draw_disease_contours(img, disease_masks)
    overlay_path = image_path.replace(".", "_overlay.")
    cv2.imwrite(overlay_path, overlay_img)

    return {
        "detections": detections,
        "infected_regions": total_regions,
        "infected_leaf_percent": infected_leaf_percent,
        "severity": severity,
        "severity_reason": reason,
        "primary_disease": primary_disease,
        "overlay_image": overlay_path,
        "all_diseases": [
            {
                "name": k,
                "regions": v["regions"],
                "infected_area_percent": round(
                    (v["pixels"] / image_area) * 100, 2
                )
            }
            for k, v in disease_stats.items()
        ],
        "recommendation": recommendation
    }
