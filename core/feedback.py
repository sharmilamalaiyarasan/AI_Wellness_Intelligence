"""
Feedback Module
Converts raw analysis output into user-friendly display data.
"""
from typing import Any

# ─── Per-exercise tips shown in the home page ─────────────────────────────────
EXERCISE_TIPS: dict[str, list[str]] = {
    "Squat": [
        "Keep your back straight throughout the movement.",
        "Push knees outward, aligned with your toes.",
        "Drive through your heels when standing back up.",
        "Go lower to maximise muscle activation.",
        "Keep your chest up and head neutral.",
    ],
    "Plank": [
        "Keep your hips level — don't sag or pike.",
        "Look down at the floor to keep your neck neutral.",
        "Squeeze glutes and abs the entire time.",
        "Breathe steadily — don't hold your breath.",
        "Arms should be directly under your shoulders.",
    ],
    "Warrior Pose": [
        "Front knee should be directly over your ankle.",
        "Back foot planted firmly at 45°.",
        "Reach your arms strongly overhead.",
        "Open your hips toward the front.",
        "Engage your core to protect your lower back.",
    ],
    "Tree Pose": [
        "Fix your gaze on one stationary point for balance.",
        "Press foot into inner thigh, thigh into foot.",
        "Keep hips level and squared to the front.",
        "Engage the standing leg fully — don't lock the knee.",
        "Breathe slowly and steadily.",
    ],
    "Mountain Pose": [
        "Feet hip-width apart, weight evenly distributed.",
        "Lift your toes, spread them, then place down.",
        "Lengthen your spine and open your chest.",
        "Relax shoulders away from your ears.",
        "Hold for 5–10 deep breaths.",
    ],
}

# ─── Feedback status thresholds ───────────────────────────────────────────────
GOOD_THRESHOLD    = 75.0
WARNING_THRESHOLD = 50.0


def get_feedback_display(feedback_list: list, score: float) -> dict[str, Any]:
    """
    Convert raw feedback from provide_specific_feedback() into a display dict.

    Returns a dict with keys:
        status          : 'good' | 'warning' | 'incorrect'
        emoji           : str
        color           : hex str (for text/icons)
        bg_color        : hex str (card background)
        border_color    : hex str
        title           : str
        subtitle        : str
        good_messages   : list[str]   — joints that are correct
        improve_messages: list[str]   — coaching directives
        score           : float
    """
    good_msgs    = [msg for _, typ, msg in feedback_list if typ == "good"]
    improve_msgs = [msg for _, typ, msg in feedback_list if typ == "improve"]

    if score >= GOOD_THRESHOLD:
        return {
            "status":          "good",
            "emoji":           "✅",
            "color":           "#22C55E",
            "bg_color":        "#F0FDF4",
            "border_color":    "#86EFAC",
            "title":           "GOOD FORM!",
            "subtitle":        "Keep it up — great technique!",
            "good_messages":   good_msgs[:4],
            "improve_messages": [],
            "score":           score,
        }
    elif score >= WARNING_THRESHOLD:
        return {
            "status":          "warning",
            "emoji":           "⚠️",
            "color":           "#F59E0B",
            "bg_color":        "#FFFBEB",
            "border_color":    "#FDE68A",
            "title":           "IMPROVE FORM",
            "subtitle":        "Small adjustments needed",
            "good_messages":   [],
            "improve_messages": improve_msgs[:3],
            "score":           score,
        }
    else:
        return {
            "status":          "incorrect",
            "emoji":           "❌",
            "color":           "#EF4444",
            "bg_color":        "#FEF2F2",
            "border_color":    "#FECACA",
            "title":           "INCORRECT FORM",
            "subtitle":        "Please re-adjust your position",
            "good_messages":   [],
            "improve_messages": improve_msgs[:3],
            "score":           score,
        }


def get_primary_tip(exercise_name: str, feedback_list: list) -> str:
    """Return the single most actionable tip to display on the live screen."""
    improve_msgs = [msg for _, typ, msg in feedback_list if typ == "improve"]
    if improve_msgs:
        # Extract the trailing hint from "Joint: angle. Adjust X°. <hint>"
        raw = improve_msgs[0]
        parts = raw.split(". ")
        if len(parts) >= 3:
            return parts[-1]
        return raw

    # Default static tip
    tips = {
        "Squat":        "Keep your back straight and go lower.",
        "Plank":        "Keep your core tight and body flat.",
        "Warrior Pose": "Front knee over ankle, arms strong overhead.",
        "Tree Pose":    "Find your balance point and breathe.",
        "Mountain Pose":"Stand tall, relax shoulders, breathe deeply.",
    }
    return tips.get(exercise_name, "Maintain proper form throughout.")


def format_duration(seconds: int) -> str:
    """Convert seconds to MM:SS string."""
    return f"{seconds // 60:02d}:{seconds % 60:02d}"
