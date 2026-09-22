"""
Pose Detection Module
Wraps MediaPipe Tasks PoseLandmarker for real-time use in Streamlit.
API mirrors exactly what the notebook uses: pose_landmarker_heavy.task
"""
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─── MediaPipe body connections for skeleton drawing ──────────────────────────
BODY_CONNECTIONS = [
    (11, 12),              # shoulders bar
    (11, 13), (13, 15),   # left arm
    (12, 14), (14, 16),   # right arm
    (15, 17), (15, 19), (17, 19),  # left hand
    (16, 18), (16, 20), (18, 20),  # right hand
    (11, 23), (12, 24),   # torso sides
    (23, 24),              # hip bar
    (23, 25), (25, 27),   # left upper/lower leg
    (24, 26), (26, 28),   # right upper/lower leg
    (27, 29), (27, 31), (29, 31),  # left foot
    (28, 30), (28, 32), (30, 32),  # right foot
]

# Landmarks to render larger (key joints for the exercises)
KEY_LANDMARKS = {11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}

# Color palette
COLOR_PURPLE  = (108, 99, 255)
COLOR_GREEN   = (34, 197, 94)
COLOR_YELLOW  = (251, 191, 36)
COLOR_RED     = (239, 68, 68)
COLOR_WHITE   = (255, 255, 255)
COLOR_DARK    = (30, 41, 59)


def find_model_path() -> str | None:
    """Locate pose_landmarker_heavy.task searching multiple common paths."""
    search_roots = [
        os.path.dirname(__file__),              # core/
        os.path.join(os.path.dirname(__file__), '..'),          # ai_fitness_coach/
        os.path.join(os.path.dirname(__file__), '..', '..'),    # fitness/
        os.path.join(os.path.dirname(__file__), '..', 'models'),
        r'c:\Users\HP\fitness',
    ]
    filename = 'pose_landmarker_heavy.task'
    for root in search_roots:
        candidate = os.path.abspath(os.path.join(root, filename))
        if os.path.exists(candidate):
            return candidate
    return None


# Module-level singleton so the heavy model loads only once
_detector = None


def get_detector():
    """Return the cached PoseLandmarker detector, creating it on first call."""
    global _detector
    if _detector is None:
        model_path = find_model_path()
        if model_path is None:
            raise FileNotFoundError(
                "pose_landmarker_heavy.task not found.\n"
                "Expected location: c:\\Users\\HP\\fitness\\pose_landmarker_heavy.task"
            )
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,   # IMAGE mode = per-frame, no timestamp needed
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _detector = vision.PoseLandmarker.create_from_options(options)
    return _detector


def detect_pose(rgb_frame: np.ndarray):
    """
    Run pose detection on a single RGB frame.
    Returns the MediaPipe detection_result object (same as notebook's `detector.detect()`).
    """
    detector = get_detector()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return detector.detect(mp_image)


def extract_keypoints(results):
    """
    Extract pose landmarks from detection results.
    EXACT copy from notebook: returns results.pose_landmarks[0] or []
    """
    if results and results.pose_landmarks and len(results.pose_landmarks) > 0:
        return results.pose_landmarks[0]
    return []


def draw_landmarks_on_frame(frame: np.ndarray, landmarks, form_score: float = None) -> np.ndarray:
    """
    Draw the pose skeleton + keypoints on an RGB frame.
    Connection color adapts to form_score (green=good, yellow=fair, red=bad).
    Returns annotated RGB frame.
    """
    if not landmarks:
        return frame

    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Choose skeleton color based on form quality
    if form_score is None:
        conn_color = COLOR_PURPLE
    elif form_score >= 70:
        conn_color = COLOR_GREEN
    elif form_score >= 45:
        conn_color = COLOR_YELLOW
    else:
        conn_color = COLOR_RED

    # ── draw connections ──────────────────────────────────────────────────────
    for start_idx, end_idx in BODY_CONNECTIONS:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue
        s = landmarks[start_idx]
        e = landmarks[end_idx]
        if s.visibility < 0.25 or e.visibility < 0.25:
            continue
        sx, sy = int(s.x * w), int(s.y * h)
        ex, ey = int(e.x * w), int(e.y * h)
        cv2.line(annotated, (sx, sy), (ex, ey), conn_color, 3, cv2.LINE_AA)

    # ── draw landmark dots ────────────────────────────────────────────────────
    for idx, lm in enumerate(landmarks):
        if lm.visibility < 0.25:
            continue
        x, y = int(lm.x * w), int(lm.y * h)
        radius = 8 if idx in KEY_LANDMARKS else 4
        cv2.circle(annotated, (x, y), radius, conn_color, -1, cv2.LINE_AA)
        cv2.circle(annotated, (x, y), radius + 2, COLOR_WHITE, 2, cv2.LINE_AA)

    return annotated


def overlay_angle_text(frame: np.ndarray, landmarks, angle_value: float,
                        joint_idx: int, label: str) -> np.ndarray:
    """Overlay joint angle text near the specified landmark on the frame."""
    if not landmarks or joint_idx >= len(landmarks):
        return frame
    h, w = frame.shape[:2]
    lm = landmarks[joint_idx]
    if lm.visibility < 0.3:
        return frame
    x = int(lm.x * w)
    y = int(lm.y * h)
    # Semi-transparent dark pill background
    text = f"{label}: {angle_value:.0f}deg"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.55, 2)
    pad = 6
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - pad, y - th - pad - 4), (x + tw + pad, y + pad), COLOR_DARK, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, text, (x, y - pad), font, 0.55, COLOR_WHITE, 2, cv2.LINE_AA)
    return frame


def get_webcam(index: int = 0) -> cv2.VideoCapture:
    """Open webcam and return VideoCapture object."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)  # CAP_DSHOW for Windows
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)  # Fallback without backend hint
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap
