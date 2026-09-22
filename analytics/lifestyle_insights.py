"""
Lifestyle Analytics & Insights Module
=====================================
Provides pure, reusable statistical analytics functions on lifestyle and wellness metrics.
Returns structured dictionaries and pandas DataFrames without hardcoded values.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


def get_average_steps(df: pd.DataFrame) -> float:
    """Calculate average daily steps across the dataset."""
    return float(round(df["Steps_Taken"].mean(), 1))


def get_average_active_minutes(df: pd.DataFrame) -> float:
    """Calculate average daily active minutes."""
    return float(round(df["Active_Minutes"].mean(), 1))


def get_average_sleep(df: pd.DataFrame) -> float:
    """Calculate average daily hours slept."""
    return float(round(df["Hours_Slept"].mean(), 2))


def get_average_calories(df: pd.DataFrame) -> float:
    """Calculate average daily calories burned."""
    return float(round(df["Calories_Burned"].mean(), 1))


def get_average_water_intake(df: pd.DataFrame) -> float:
    """Calculate average daily water intake in Liters."""
    return float(round(df["Water_Intake (Liters)"].mean(), 2))


def get_average_heart_rate(df: pd.DataFrame) -> float:
    """Calculate average resting heart rate (bpm)."""
    return float(round(df["Heart_Rate (bpm)"].mean(), 1))


def get_average_stress(df: pd.DataFrame) -> float:
    """Calculate average stress level on a 1–10 scale."""
    return float(round(df["Stress_Level (1-10)"].mean(), 2))


def get_workout_rest_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Return distribution count of each workout type (Yoga, Cardio, Strength, Rest)."""
    return df["Workout_Type"].value_counts(dropna=False).to_dict()


def get_mood_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """Return distribution count of daily mood states."""
    return df["Mood"].value_counts(dropna=False).to_dict()


def get_activity_vs_sleep_relationship(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the association between physical activity (steps) and sleep duration.
    Returns correlation coefficient, p-value proxy, and binned mean sleep by step quartile.
    """
    corr = float(df["Steps_Taken"].corr(df["Hours_Slept"]))
    df_temp = df.copy()
    df_temp["Step_Bin"] = pd.qcut(df_temp["Steps_Taken"], q=4, labels=["Low (Q1)", "Moderate (Q2)", "High (Q3)", "Very High (Q4)"])
    binned_sleep = df_temp.groupby("Step_Bin", observed=False)["Hours_Slept"].agg(["mean", "std", "count"]).round(2)

    return {
        "correlation": round(corr, 4),
        "interpretation": "Observed correlation between daily step volume and hours slept.",
        "binned_stats": binned_sleep.to_dict(orient="index"),
    }


def get_activity_vs_calories_relationship(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the association between daily active minutes and calories burned.
    Returns Pearson correlation and calories burned by active minute tiers.
    """
    corr = float(df["Active_Minutes"].corr(df["Calories_Burned"]))
    df_temp = df.copy()
    df_temp["Active_Tier"] = pd.cut(
        df_temp["Active_Minutes"],
        bins=[0, 45, 75, 105, 150],
        labels=["<45 mins", "45–75 mins", "75–105 mins", ">105 mins"]
    )
    binned_cals = df_temp.groupby("Active_Tier", observed=False)["Calories_Burned"].agg(["mean", "std"]).round(1)

    return {
        "correlation": round(corr, 4),
        "interpretation": "Observed correlation between duration of active minutes and calories burned.",
        "tier_stats": binned_cals.to_dict(orient="index"),
    }


def get_stress_vs_sleep_relationship(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the relationship between daily stress level (1–10) and sleep duration.
    Returns correlation coefficient and mean sleep hours per stress level.
    """
    corr = float(df["Stress_Level (1-10)"].corr(df["Hours_Slept"]))
    stress_sleep = df.groupby("Stress_Level (1-10)")["Hours_Slept"].agg(["mean", "count"]).round(2)

    return {
        "correlation": round(corr, 4),
        "interpretation": "Observed correlation between perceived stress rating and sleep duration.",
        "sleep_by_stress": stress_sleep.to_dict(orient="index"),
    }


def get_weekend_vs_weekday_behaviour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare lifestyle metrics between Weekdays (Monday–Friday) and Weekends (Saturday–Sunday).
    Returns a summary DataFrame with mean metrics and absolute differences.
    """
    metrics = [
        "Steps_Taken", "Active_Minutes", "Hours_Slept",
        "Calories_Burned", "Water_Intake (Liters)",
        "Heart_Rate (bpm)", "Stress_Level (1-10)"
    ]
    if "Wellness_Score" in df.columns:
        metrics.append("Wellness_Score")

    df_temp = df.copy()
    if "Is_Weekend" not in df_temp.columns:
        df_temp["Date"] = pd.to_datetime(df_temp["Date"])
        df_temp["Is_Weekend"] = df_temp["Date"].dt.dayofweek.isin([5, 6]).astype(int)

    grouped = df_temp.groupby("Is_Weekend")[metrics].mean().round(2)
    grouped.index = ["Weekday (Mon–Fri)", "Weekend (Sat–Sun)"]
    summary = grouped.T
    summary["Diff (Weekend - Weekday)"] = (summary["Weekend (Sat–Sun)"] - summary["Weekday (Mon–Fri)"]).round(2)
    summary["Pct_Diff (%)"] = ((summary["Diff (Weekend - Weekday)"] / summary["Weekday (Mon–Fri)"]) * 100.0).round(1)

    return summary


def get_monthly_activity_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly average steps, active minutes, and calories burned."""
    df_temp = df.copy()
    df_temp["Date"] = pd.to_datetime(df_temp["Date"])
    df_temp["Month"] = df_temp["Date"].dt.strftime("%Y-%m")

    monthly = df_temp.groupby("Month")[["Steps_Taken", "Active_Minutes", "Calories_Burned"]].mean().round(1)
    return monthly


def get_monthly_sleep_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly average sleep duration and sleep score."""
    df_temp = df.copy()
    df_temp["Date"] = pd.to_datetime(df_temp["Date"])
    df_temp["Month"] = df_temp["Date"].dt.strftime("%Y-%m")

    cols = ["Hours_Slept"]
    if "Sleep_Score" in df_temp.columns:
        cols.append("Sleep_Score")

    monthly = df_temp.groupby("Month")[cols].mean().round(2)
    return monthly


def get_monthly_wellness_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly trends across overall Wellness Score and individual sub-scores."""
    df_temp = df.copy()
    df_temp["Date"] = pd.to_datetime(df_temp["Date"])
    df_temp["Month"] = df_temp["Date"].dt.strftime("%Y-%m")

    score_cols = [c for c in df_temp.columns if "Score" in c]
    if not score_cols:
        from analytics.wellness_score import calculate_wellness_score
        df_temp = calculate_wellness_score(df_temp)
        score_cols = [c for c in df_temp.columns if "Score" in c]

    monthly = df_temp.groupby("Month")[score_cols].mean().round(2)
    return monthly


def get_user_level_wellness_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 365-day tracking data by User_ID.
    Computes per-user means for activity, recovery, vitals, workout frequency, and wellness scores.
    """
    df_temp = df.copy()
    if "Wellness_Score" not in df_temp.columns:
        from analytics.wellness_score import calculate_wellness_score
        df_temp = calculate_wellness_score(df_temp)

    # Workout frequency (fraction of days with active workout vs Rest)
    df_temp["Has_Workout"] = (df_temp["Workout_Type"] != "Rest").astype(int)

    user_agg = df_temp.groupby("User_ID").agg(
        Full_Name=("Full Name", "first"),
        Age=("Age", "first"),
        Gender=("Gender", "first"),
        Height_cm=("Height (cm)", "first"),
        Weight_kg=("Weight (kg)", "first"),
        BMI=("BMI", "first") if "BMI" in df_temp.columns else ("Weight (kg)", lambda w: 0.0),
        Avg_Steps=("Steps_Taken", "mean"),
        Avg_Active_Mins=("Active_Minutes", "mean"),
        Avg_Sleep_Hours=("Hours_Slept", "mean"),
        Avg_Calories=("Calories_Burned", "mean"),
        Avg_Water_Liters=("Water_Intake (Liters)", "mean"),
        Avg_Heart_Rate=("Heart_Rate (bpm)", "mean"),
        Avg_Stress=("Stress_Level (1-10)", "mean"),
        Workout_Frequency=("Has_Workout", "mean"),
        Avg_Sleep_Score=("Sleep_Score", "mean"),
        Avg_Activity_Score=("Activity_Score", "mean"),
        Avg_Hydration_Score=("Hydration_Score", "mean"),
        Avg_Stress_Score=("Stress_Score", "mean"),
        Avg_Heart_Rate_Score=("Heart_Rate_Score", "mean"),
        Avg_Wellness_Score=("Wellness_Score", "mean"),
    ).round(2)

    return user_agg


def get_lifestyle_summary_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Return key headline indicators in a structured dictionary."""
    return {
        "avg_steps": get_average_steps(df),
        "avg_active_minutes": get_average_active_minutes(df),
        "avg_sleep_hours": get_average_sleep(df),
        "avg_calories": get_average_calories(df),
        "avg_water_liters": get_average_water_intake(df),
        "avg_heart_rate": get_average_heart_rate(df),
        "avg_stress_level": get_average_stress(df),
        "workout_distribution": get_workout_rest_distribution(df),
        "mood_distribution": get_mood_distribution(df),
    }
