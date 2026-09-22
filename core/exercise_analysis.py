"""
Exercise Analysis Module
Uses the trained Random Forest Classifier (models/yoga_pose_classifier.pkl)
and real dataset posture reference profiles (models/dataset_pose_references.json)
extracted directly from your Kaggle Yoga Posture Dataset.

Contains EXACT copies of all core math/feedback functions from notebook:
  - calculate_angle()
  - calculate_angles()
  - calculate_adjustments()
  - calculate_percentage()
  - provide_specific_feedback()
  - calculate_correctness_percentage()
"""

import os
import json
import pickle
import numpy as np

# ─── Body part labels (from notebook) ────────────────────────────────────────
BODY_PARTS = [
    "Left Elbow", "Right Elbow",
    "Left Shoulder", "Right Shoulder",
    "Left Hip", "Right Hip",
    "Left Knee", "Right Knee",
]

# Angle vector indices
IDX_L_ELBOW    = 0
IDX_R_ELBOW    = 1
IDX_L_SHOULDER = 2
IDX_R_SHOULDER = 3
IDX_L_HIP      = 4
IDX_R_HIP      = 5
IDX_L_KNEE     = 6
IDX_R_KNEE     = 7

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "yoga_pose_classifier.pkl")
REF_JSON_PATH = os.path.join(MODEL_DIR, "dataset_pose_references.json")

# ─── Load trained Random Forest model & dataset pose reference profiles ────────
_classifier = None
_dataset_references = {}


def load_dataset_model():
    global _classifier, _dataset_references
    if _classifier is None and os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                _classifier = pickle.load(f)
        except Exception as e:
            print(f"Error loading trained classifier: {e}")

    if not _dataset_references and os.path.exists(REF_JSON_PATH):
        try:
            with open(REF_JSON_PATH, "r") as f:
                _dataset_references = json.load(f)
        except Exception as e:
            print(f"Error loading pose reference profiles: {e}")

    return _classifier, _dataset_references


# Load at import time
load_dataset_model()

# User-friendly display names for dataset pose classes
POSE_DISPLAY_NAMES = {
    "Adho Mukha Svanasana": "Adho Mukha Svanasana (Downward-Facing Dog)",
    "Adho Mukha Vrksasana": "Adho Mukha Vrksasana (Handstand)",
    "Anjaneyasana":         "Anjaneyasana (Low Lunge)",
    "Ardha Chandrasana":    "Ardha Chandrasana (Half Moon Pose)",
    "Baddha Konasana":       "Baddha Konasana (Bound Angle Pose)",
    "Bakasana":             "Bakasana (Crow Pose)",
    "Balasana":             "Balasana (Child's Pose)",
    "Bhujangasana":         "Bhujangasana (Cobra Pose)",
    "Garudasana":           "Garudasana (Eagle Pose)",
    "Halasana":             "Halasana (Plow Pose)",
    "Malasana":             "Malasana (Garland Pose)",
    "Navasana":             "Navasana (Boat Pose)",
    "Padmasana":            "Padmasana (Lotus Pose)",
    "Phalakasana":          "Phalakasana (Plank Pose)",
    "Salamba Bhujangasana": "Salamba Bhujangasana (Sphinx Pose)",
    "Setu Bandha Sarvangasana": "Setu Bandha (Bridge Pose)",
    "Sivasana":             "Sivasana (Corpse Pose)",
    "Trikonasana":          "Trikonasana (Triangle Pose)",
    "Urdhva Dhanurasana":   "Urdhva Dhanurasana (Wheel Pose)",
    "Ustrasana":            "Ustrasana (Camel Pose)",
    "Utkatasana":           "Utkatasana (Chair Pose / Squat)",
    "Uttanasana":           "Uttanasana (Standing Forward Bend)",
    "Virabhadrasana One":   "Virabhadrasana I (Warrior 1)",
    "Virabhadrasana Two":   "Virabhadrasana II (Warrior 2)",
    "Virabhadrasana Three": "Virabhadrasana III (Warrior 3)",
    "Vrksasana":            "Vrksasana (Tree Pose)",
}

# Supported exercises list from dataset
SUPPORTED_EXERCISES = list(_dataset_references.keys()) if _dataset_references else [
    "Utkatasana", "Phalakasana", "Virabhadrasana One", "Vrksasana", "Adho Mukha Svanasana", "Anjaneyasana"
]

# Build metadata dynamically for dataset pose categories
EXERCISE_INFO = {}
for pose in SUPPORTED_EXERCISES:
    disp = POSE_DISPLAY_NAMES.get(pose, pose)
    EXERCISE_INFO[pose] = {
        "name": pose,
        "display_name": disp,
        "category": "Dataset Pose",
        "description": f"Real Kaggle Dataset pose class: {disp}",
        "icon": "🧘",
        "rep_type": "reps" if pose in ["Utkatasana", "Uttanasana"] else "time",
        "primary_angle_idx": IDX_R_KNEE if "Utk" in pose else IDX_L_HIP,
        "primary_angle_label": "Knee Angle" if "Utk" in pose else "Hip Angle",
        "down_threshold": 120.0 if pose == "Utkatasana" else None,
        "up_threshold":  160.0 if pose == "Utkatasana" else None,
        "sets": "3 Sets × 12 Reps" if pose in ["Utkatasana"] else "3 Sets × 30 sec",
    }


# ─── EXACT COPIES FROM NOTEBOOK ──────────────────────────────────────────────

def calculate_angle(a, b, c) -> float:
    """EXACT copy from notebook."""
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return float(np.degrees(angle))


def calculate_angles(keypoints) -> list:
    """EXACT copy from notebook."""
    angles = []
    if keypoints:
        left_elbow_angle    = calculate_angle(keypoints[11], keypoints[13], keypoints[15])
        right_elbow_angle   = calculate_angle(keypoints[12], keypoints[14], keypoints[16])
        left_shoulder_angle  = calculate_angle(keypoints[23], keypoints[11], keypoints[13])
        right_shoulder_angle = calculate_angle(keypoints[24], keypoints[12], keypoints[14])
        left_hip_angle       = calculate_angle(keypoints[25], keypoints[23], keypoints[11])
        right_hip_angle      = calculate_angle(keypoints[26], keypoints[24], keypoints[12])
        left_knee_angle      = calculate_angle(keypoints[23], keypoints[25], keypoints[27])
        right_knee_angle     = calculate_angle(keypoints[24], keypoints[26], keypoints[28])
        angles = [
            left_elbow_angle, right_elbow_angle,
            left_shoulder_angle, right_shoulder_angle,
            left_hip_angle, right_hip_angle,
            left_knee_angle, right_knee_angle,
        ]
    return angles


def calculate_adjustments(correct_angles, user_angles) -> np.ndarray:
    """EXACT copy from notebook."""
    correct_angles = np.array([a for a in correct_angles], dtype=float)
    user_angles    = np.array([a for a in user_angles],    dtype=float)
    return correct_angles - user_angles


def calculate_percentage(adjustments, lower_threshold: float = 10.0,
                          upper_threshold: float = 50.0) -> float:
    """EXACT copy from notebook."""
    angular_distances = np.abs(adjustments)
    correctness_percentages = np.zeros_like(angular_distances, dtype=float)

    below = angular_distances <= lower_threshold
    correctness_percentages[below] = 100.0

    between = (angular_distances > lower_threshold) & (angular_distances < upper_threshold)
    correctness_percentages[between] = 100.0 * (
        1.0 - (angular_distances[between] - lower_threshold) / (upper_threshold - lower_threshold)
    )

    return float(np.mean(correctness_percentages))


def provide_specific_feedback(correct_pose_angles, user_pose_angles,
                               model=None, threshold: float = 5.0):
    """EXACT copy from notebook."""
    user_pose_angles    = np.array([user_pose_angles])
    correct_pose_angles = np.array([correct_pose_angles])
    adjustments = calculate_adjustments(user_pose_angles, correct_pose_angles)
    score = calculate_percentage(adjustments)

    feedback = []
    for idx, (correct_angle, user_angle, adjustment) in enumerate(
        zip(correct_pose_angles[0], user_pose_angles[0], adjustments[0])
    ):
        body_part = BODY_PARTS[idx]
        if abs(adjustment) < threshold:
            feedback.append((body_part, "good", f"{body_part}: Good alignment!"))
        else:
            direction = "decrease" if adjustment > 0 else "increase"
            hint = _direction_hint(body_part, direction)
            msg = (
                f"{body_part}: {user_angle:.1f}° "
                f"(adjust {abs(adjustment):.1f}° to {direction}). {hint}"
            )
            feedback.append((body_part, "improve", msg))

    return feedback, score


def _direction_hint(body_part: str, direction: str) -> str:
    """Directional coaching hint. From notebook logic."""
    table = {
        ("Left Elbow",     "increase"): "Lift your arm higher.",
        ("Left Elbow",     "decrease"): "Lower your arm.",
        ("Right Elbow",    "increase"): "Lift your arm higher.",
        ("Right Elbow",    "decrease"): "Lower your arm.",
        ("Left Shoulder",  "increase"): "Move your shoulder up.",
        ("Left Shoulder",  "decrease"): "Move your shoulder down.",
        ("Right Shoulder", "increase"): "Move your shoulder up.",
        ("Right Shoulder", "decrease"): "Move your shoulder down.",
        ("Left Hip",       "increase"): "Lift your hip higher.",
        ("Left Hip",       "decrease"): "Lower your hip / sit deeper.",
        ("Right Hip",      "increase"): "Lift your hip higher.",
        ("Right Hip",      "decrease"): "Lower your hip / sit deeper.",
        ("Left Knee",      "increase"): "Extend your knee more.",
        ("Left Knee",      "decrease"): "Bend your knee deeper.",
        ("Right Knee",     "increase"): "Extend your knee more.",
        ("Right Knee",     "decrease"): "Bend your knee deeper.",
    }
    return table.get((body_part, direction), "Adjust your position.")


def calculate_correctness_percentage(correct_angles, user_angles,
                                      lower_threshold: float = 10.0,
                                      upper_threshold: float = 50.0) -> float:
    """EXACT copy from notebook."""
    correct_angles = np.array(correct_angles)
    user_angles    = np.array(user_angles)
    angular_distances = np.abs(calculate_adjustments(correct_angles, user_angles))
    correctness_percentages = np.zeros_like(angular_distances, dtype=float)

    below = angular_distances <= lower_threshold
    correctness_percentages[below] = 100.0

    between = (angular_distances > lower_threshold) & (angular_distances < upper_threshold)
    correctness_percentages[between] = 100.0 * (
        1.0 - (angular_distances[between] - lower_threshold) / (upper_threshold - lower_threshold)
    )

    return float(np.mean(correctness_percentages))


# ─── Trained Dataset Model Inference ──────────────────────────────────────────

def predict_exercise_from_angles(user_angles: list) -> tuple[str, float]:
    """
    Predict pose category using the trained Random Forest Classifier
    and compute match score against the dataset reference angles.
    """
    clf, ref_dict = load_dataset_model()
    if not user_angles or len(user_angles) < 8:
        return "Unknown Pose", 0.0

    user_vec = np.array(user_angles).reshape(1, -1)

    predicted_class = "Utkatasana"
    if clf is not None:
        try:
            predicted_class = clf.predict(user_vec)[0]
        except Exception:
            pass

    # Fallback/validation using dataset reference profiles
    if ref_dict and predicted_class in ref_dict:
        ref_angles = ref_dict[predicted_class]
        score = calculate_correctness_percentage(ref_angles, user_angles)
    else:
        # Match against all references to find top match
        best_cls = predicted_class
        max_score = 0.0
        for cls_name, ref_angles in ref_dict.items():
            s = calculate_correctness_percentage(ref_angles, user_angles)
            if s > max_score:
                max_score = s
                best_cls = cls_name
        predicted_class = best_cls
        score = max_score

    return predicted_class, round(score, 1)


def get_reference_angles(exercise_name: str) -> list:
    """Return reference angle profile for a given exercise from trained dataset."""
    _, ref_dict = load_dataset_model()
    if ref_dict and exercise_name in ref_dict:
        return ref_dict[exercise_name]
    return [175.0] * 8


def get_primary_angle(user_angles: list, exercise_name: str):
    info = EXERCISE_INFO.get(exercise_name, {})
    idx  = info.get("primary_angle_idx", IDX_R_KNEE)
    if user_angles and idx < len(user_angles):
        return user_angles[idx]
    return None
