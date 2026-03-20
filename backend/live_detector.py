from collections import defaultdict, deque
from pesticide_data import get_pesticide_recommendation

CONF_TH = 0.32
FRAME_RATIO_TH = 0.02
SMOOTH_ALPHA = 0.6


class LiveSession:
    def __init__(self):
        self.total_frames = 0
        self.leaf_frames = 0
        self.history = deque(maxlen=300)

        self.stats = defaultdict(lambda: {
            "count": 0,
            "conf_sum": 0.0,
            "smooth_conf": 0.0,
            "streak": 0,
            "max_streak": 0
        })

    # ==========================================
    # UPDATE PER FRAME
    # ==========================================
    def update(self, detections):
        self.total_frames += 1
        self.history.append(detections)

        detected_classes = set()
        leaf_detected = False

        for d in detections:
            cls = d["class"]
            conf = d["conf"]

            detected_classes.add(cls)

            if "leaf" in cls.lower():
                leaf_detected = True

            s = self.stats[cls]
            s["count"] += 1
            s["conf_sum"] += conf

            # smoothing
            if s["smooth_conf"] == 0:
                s["smooth_conf"] = conf
            else:
                s["smooth_conf"] = (
                    SMOOTH_ALPHA * conf +
                    (1 - SMOOTH_ALPHA) * s["smooth_conf"]
                )

            s["streak"] += 1
            s["max_streak"] = max(s["max_streak"], s["streak"])

        if leaf_detected:
            self.leaf_frames += 1

        for cls in self.stats:
            if cls not in detected_classes:
                self.stats[cls]["streak"] = 0

    def reset(self):
        self.total_frames = 0
        self.leaf_frames = 0
        self.history.clear()
        self.stats.clear()

    # ==========================================
    # SUMMARIZE
    # ==========================================
    def summarize(self):

        if self.leaf_frames < max(3, self.total_frames * 0.1):
            return {
                "primary_disease": None,
                "severity": "NO_LEAF_VISIBLE",
                "infection_ratio": 0,
                "avg_confidence": 0,
                "infected_leaves": 0,
                "total_leaves": self.total_frames,
                "recommendation": {
                    "type": "invalid",
                    "message": "No clear leaf detected."
                }
            }

        valid = {}

        for cls, data in self.stats.items():
            count = data["count"]
            if count == 0:
                continue

            ratio = count / max(self.total_frames, 1)
            avg_conf = data["conf_sum"] / count
            smooth_conf = data["smooth_conf"]

            streak_bonus = min(data["max_streak"] / 20, 1.0)

            score = (
                smooth_conf * 0.4 +
                avg_conf * 0.3 +
                ratio * 0.2 +
                streak_bonus * 0.1
            )

            if ratio >= FRAME_RATIO_TH and score >= CONF_TH:
                valid[cls] = (ratio, score, count)

        # ===============================
        # HEALTHY SESSION
        # ===============================
        if not valid:
            return {
                "primary_disease": None,
                "severity": "NONE",
                "infection_ratio": 0,
                "avg_confidence": 0,
                "infected_leaves": 0,
                "total_leaves": self.total_frames,
                "recommendation": {
                    "type": "healthy",
                    "message": "No disease detected"
                }
            }

        primary = max(valid, key=lambda k: valid[k][1])
        ratio, score, infected_count = valid[primary]

        # ⭐ CRITICAL FIX
        # healthy leaf → infected = 0
        if "healthy" in primary.lower():
            infected_count = 0
            severity = "NONE"
        else:
            severity = (
                "HIGH" if ratio > 0.35 else
                "MEDIUM" if ratio > 0.12 else
                "LOW"
            )

        return {
            "primary_disease": primary,
            "severity": severity,
            "infection_ratio": round(ratio * 100, 2),
            "avg_confidence": round(score * 100, 2),
            "infected_leaves": infected_count,
            "total_leaves": self.total_frames,
            "recommendation": get_pesticide_recommendation(primary, severity)
        }
