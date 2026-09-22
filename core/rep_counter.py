"""
Rep Counter Module
Tracks exercise repetitions using joint angle state machine.
Works for rep-based exercises (e.g. Squat) and time-based holds (e.g. Plank).
"""
import time


class RepCounter:
    """
    Angle-threshold state machine for counting exercise repetitions.

    For rep-based exercises:
        Stage transitions: 'up' → 'down' (angle crosses down_threshold) → rep counted
        A rep is recorded when the user goes DOWN (angle < down_threshold)
        while previously in 'up' position (angle > up_threshold).

    For time-based exercises:
        Tracks elapsed hold time instead of discrete reps.
    """

    def __init__(self, exercise_name: str, exercise_info: dict):
        self.exercise    = exercise_name
        self.info        = exercise_info
        self.rep_type    = exercise_info.get("rep_type", "reps")

        # Rep tracking
        self.reps          = 0
        self.correct_reps  = 0
        self.incorrect_reps = 0
        self.stage         = "up"          # current phase: 'up' or 'down'

        # Collect form scores within the current rep to average later
        self._rep_scores: list[float] = []

        # Time tracking
        self.start_time = time.time()

    # ── public API ────────────────────────────────────────────────────────────

    def update(self, user_angles: list, form_score: float) -> None:
        """
        Call once per processed frame.
        Updates rep count and correct/incorrect tally.
        """
        if self.rep_type != "reps" or not user_angles:
            return

        primary_idx     = self.info.get("primary_angle_idx", 7)
        down_threshold  = self.info.get("down_threshold")
        up_threshold    = self.info.get("up_threshold")

        if down_threshold is None or up_threshold is None:
            return
        if primary_idx >= len(user_angles):
            return

        angle = user_angles[primary_idx]
        self._rep_scores.append(form_score)

        # ── state machine ─────────────────────────────────────────────────────
        if angle > up_threshold:
            # Back to "up" position — reset stage
            self.stage = "up"

        if angle < down_threshold and self.stage == "up":
            # Crossed into "down" → count the rep
            self.stage = "down"
            self.reps += 1
            avg_score = (
                sum(self._rep_scores) / len(self._rep_scores)
                if self._rep_scores else 0.0
            )
            if avg_score >= 65.0:
                self.correct_reps += 1
            else:
                self.incorrect_reps += 1
            self._rep_scores = []   # reset for next rep

    @property
    def elapsed_seconds(self) -> int:
        return int(time.time() - self.start_time)

    @property
    def elapsed_str(self) -> str:
        t = self.elapsed_seconds
        return f"{t // 60:02d}:{t % 60:02d}"

    def form_accuracy(self) -> float:
        """Compute overall form accuracy % from completed reps."""
        if self.reps == 0:
            return 0.0
        return round(self.correct_reps / self.reps * 100, 1)

    def get_summary(self) -> dict:
        return {
            "exercise":         self.exercise,
            "total_reps":       self.reps,
            "correct_reps":     self.correct_reps,
            "incorrect_reps":   self.incorrect_reps,
            "duration_seconds": self.elapsed_seconds,
            "duration_str":     self.elapsed_str,
            "form_accuracy":    self.form_accuracy(),
            "rep_type":         self.rep_type,
        }
