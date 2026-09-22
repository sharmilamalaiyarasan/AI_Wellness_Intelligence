"""
Wellness Score Calculation Module
=================================
Calculates continuous, objective sub-scores and composite daily Wellness Scores (0–100)
grounded in empirical data and established wellness guidelines.
See data/wellness_score_design.md for full specification and methodology.
"""

from typing import Union
import numpy as np
import pandas as pd

# ── Dimension Weights ─────────────────────────────────────────────────────────
WEIGHT_SLEEP: float = 0.25
WEIGHT_ACTIVITY: float = 0.25
WEIGHT_STRESS: float = 0.20
WEIGHT_HYDRATION: float = 0.15
WEIGHT_HEART_RATE: float = 0.15


def calculate_sleep_score(hours_slept: Union[float, pd.Series, np.ndarray]) -> Union[float, pd.Series, np.ndarray]:
    """
    Calculate sleep restoration score (0–100).
    Optimal target: >= 7.0 hours/night (National Sleep Foundation guideline).
    Penalizes insufficient sleep linearly down to 0 at <= 4.0 hours (dataset minimum).
    """
    if isinstance(hours_slept, (pd.Series, np.ndarray)):
        score = np.where(
            hours_slept >= 7.0,
            100.0,
            np.clip((hours_slept - 4.0) / (7.0 - 4.0) * 100.0, 0.0, 100.0)
        )
        return pd.Series(score, index=hours_slept.index) if isinstance(hours_slept, pd.Series) else score
    else:
        if hours_slept >= 7.0:
            return 100.0
        return float(np.clip((hours_slept - 4.0) / (7.0 - 4.0) * 100.0, 0.0, 100.0))


def calculate_activity_score(
    steps: Union[float, pd.Series, np.ndarray],
    active_minutes: Union[float, pd.Series, np.ndarray]
) -> Union[float, pd.Series, np.ndarray]:
    """
    Calculate physical activity score (0–100) blending volume (steps) and duration (active minutes).
    - Steps component (50%): 2,000 steps (min) -> 0; >= 10,000 steps -> 100.
    - Active minutes component (50%): 30 mins -> 0; >= 60 mins -> 100 (WHO daily guideline).
    """
    step_score = np.clip((steps - 2000.0) / (10000.0 - 2000.0) * 100.0, 0.0, 100.0)
    min_score = np.clip((active_minutes - 30.0) / (60.0 - 30.0) * 100.0, 0.0, 100.0)
    composite = 0.5 * step_score + 0.5 * min_score
    return composite


def calculate_hydration_score(
    water_intake_liters: Union[float, pd.Series, np.ndarray]
) -> Union[float, pd.Series, np.ndarray]:
    """
    Calculate daily hydration score (0–100).
    Optimal target: >= 3.0 Liters/day.
    Penalizes low fluid intake linearly down to 0 at <= 1.5 Liters (dataset minimum).
    """
    if isinstance(water_intake_liters, (pd.Series, np.ndarray)):
        score = np.where(
            water_intake_liters >= 3.0,
            100.0,
            np.clip((water_intake_liters - 1.5) / (3.0 - 1.5) * 100.0, 0.0, 100.0)
        )
        return pd.Series(score, index=water_intake_liters.index) if isinstance(water_intake_liters, pd.Series) else score
    else:
        if water_intake_liters >= 3.0:
            return 100.0
        return float(np.clip((water_intake_liters - 1.5) / (3.0 - 1.5) * 100.0, 0.0, 100.0))


def calculate_stress_score(
    stress_level: Union[float, pd.Series, np.ndarray]
) -> Union[float, pd.Series, np.ndarray]:
    """
    Calculate stress management score (0–100) via inverse linear scaling.
    Stress Level 1 (Lowest stress) -> 100.
    Stress Level 10 (Highest stress) -> 0.
    """
    return np.clip((10.0 - stress_level) / (10.0 - 1.0) * 100.0, 0.0, 100.0)


def calculate_heart_rate_score(
    heart_rate_bpm: Union[float, pd.Series, np.ndarray]
) -> Union[float, pd.Series, np.ndarray]:
    """
    Calculate resting cardiovascular efficiency score (0–100).
    AHA healthy resting heart rate baseline:
    - <= 65 bpm: 100 (Optimal athletic / resting tone)
    - 65 to 95 bpm: scales linearly from 100 down to 20
    - > 95 bpm: scaled down from 20 to 0 (elevated resting strain)
    """
    if isinstance(heart_rate_bpm, (pd.Series, np.ndarray)):
        score = np.where(
            heart_rate_bpm <= 65.0,
            100.0,
            np.clip(100.0 - ((heart_rate_bpm - 65.0) / (95.0 - 65.0)) * 80.0, 0.0, 100.0)
        )
        return pd.Series(score, index=heart_rate_bpm.index) if isinstance(heart_rate_bpm, pd.Series) else score
    else:
        if heart_rate_bpm <= 65.0:
            return 100.0
        return float(np.clip(100.0 - ((heart_rate_bpm - 65.0) / (95.0 - 65.0)) * 80.0, 0.0, 100.0))


def calculate_wellness_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 5 sub-scores and composite daily Wellness Score (0–100).
    Returns a new DataFrame containing all original columns plus:
      - Sleep_Score
      - Activity_Score
      - Hydration_Score
      - Stress_Score
      - Heart_Rate_Score
      - Wellness_Score
    """
    out = df.copy()

    # Calculate component scores
    out["Sleep_Score"] = calculate_sleep_score(out["Hours_Slept"]).round(2)
    out["Activity_Score"] = calculate_activity_score(out["Steps_Taken"], out["Active_Minutes"]).round(2)
    out["Hydration_Score"] = calculate_hydration_score(out["Water_Intake (Liters)"]).round(2)
    out["Stress_Score"] = calculate_stress_score(out["Stress_Level (1-10)"]).round(2)
    out["Heart_Rate_Score"] = calculate_heart_rate_score(out["Heart_Rate (bpm)"]).round(2)

    # Weighted convex combination
    wellness = (
        WEIGHT_SLEEP * out["Sleep_Score"] +
        WEIGHT_ACTIVITY * out["Activity_Score"] +
        WEIGHT_STRESS * out["Stress_Score"] +
        WEIGHT_HYDRATION * out["Hydration_Score"] +
        WEIGHT_HEART_RATE * out["Heart_Rate_Score"]
    )

    out["Wellness_Score"] = np.clip(wellness, 0.0, 100.0).round(2)
    return out
