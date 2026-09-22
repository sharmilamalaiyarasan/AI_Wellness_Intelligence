"""
Unit and Integration Tests for Phase 3 Lifestyle Analytics & Wellness Intelligence
=================================================================================
Tests:
  1. Wellness score range is strictly 0–100.
  2. No unexpected NaN values in scored outputs.
  3. Component scores (Sleep, Activity, Hydration, Stress, Heart Rate) are within expected [0, 100] ranges.
  4. One user gets exactly one cluster assignment.
  5. Cluster IDs and labels are valid and non-null.
  6. Required output files (daily_wellness_scores.csv, user_lifestyle_profiles.csv, 7 figures) are present.
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
import numpy as np

from analytics.wellness_score import (
    calculate_sleep_score,
    calculate_activity_score,
    calculate_hydration_score,
    calculate_stress_score,
    calculate_heart_rate_score,
    calculate_wellness_score,
)
from analytics.lifestyle_insights import (
    get_average_steps,
    get_average_sleep,
    get_workout_rest_distribution,
    get_user_level_wellness_summaries,
)
from analytics.segmentation import perform_segmentation, evaluate_cluster_range


class TestPhase3Analytics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.cleaned_csv = os.path.join(cls.base_dir, "data", "processed", "fitness_health_cleaned.csv")
        cls.daily_csv = os.path.join(cls.base_dir, "data", "processed", "daily_wellness_scores.csv")
        cls.user_csv = os.path.join(cls.base_dir, "data", "processed", "user_lifestyle_profiles.csv")
        cls.fig_dir = os.path.join(cls.base_dir, "reports", "figures")

        cls.df = pd.read_csv(cls.cleaned_csv)
        cls.scored_df = calculate_wellness_score(cls.df)
        cls.user_df = get_user_level_wellness_summaries(cls.scored_df)
        cls.segmented_df, cls.km, cls.scaler, cls.mapping = perform_segmentation(cls.user_df, n_clusters=3)

    # ── Test 1: Wellness Score Bounds ──────────────────────────────────────────
    def test_wellness_score_range(self):
        """Wellness score must be strictly bounded between 0.0 and 100.0."""
        scores = self.scored_df["Wellness_Score"]
        self.assertGreaterEqual(scores.min(), 0.0, "Wellness score has values below 0.0")
        self.assertLessEqual(scores.max(), 100.0, "Wellness score has values above 100.0")

    # ── Test 2: No Unexpected NaN Values ───────────────────────────────────────
    def test_no_nan_values(self):
        """Cleaned, daily scored, and user profile datasets must have zero NaNs."""
        self.assertEqual(self.scored_df["Wellness_Score"].isna().sum(), 0, "NaNs found in daily Wellness_Score")
        score_cols = ["Sleep_Score", "Activity_Score", "Hydration_Score", "Stress_Score", "Heart_Rate_Score"]
        for col in score_cols:
            self.assertEqual(self.scored_df[col].isna().sum(), 0, f"NaNs found in {col}")
        self.assertEqual(self.segmented_df["Cluster_ID"].isna().sum(), 0, "NaNs found in user Cluster_ID")

    # ── Test 3: Component Score Bounds & Properties ───────────────────────────
    def test_component_score_bounds(self):
        """All 5 component scores must be within [0, 100]."""
        for col in ["Sleep_Score", "Activity_Score", "Hydration_Score", "Stress_Score", "Heart_Rate_Score"]:
            s = self.scored_df[col]
            self.assertGreaterEqual(s.min(), 0.0, f"{col} has value < 0")
            self.assertLessEqual(s.max(), 100.0, f"{col} has value > 100")

    def test_boundary_edge_cases(self):
        """Test component function edge cases."""
        # Sleep
        self.assertEqual(calculate_sleep_score(8.0), 100.0)
        self.assertEqual(calculate_sleep_score(4.0), 0.0)
        self.assertEqual(calculate_sleep_score(2.0), 0.0)
        # Stress
        self.assertEqual(calculate_stress_score(1.0), 100.0)
        self.assertEqual(calculate_stress_score(10.0), 0.0)
        # Hydration
        self.assertEqual(calculate_hydration_score(3.5), 100.0)
        self.assertEqual(calculate_hydration_score(1.5), 0.0)
        # Heart rate
        self.assertEqual(calculate_heart_rate_score(60.0), 100.0)
        self.assertLess(calculate_heart_rate_score(95.0), 25.0)

    # ── Test 4: One User Gets Exactly One Cluster ──────────────────────────────
    def test_one_cluster_per_user(self):
        """Exactly 100 users, each mapped to a single unique cluster ID."""
        self.assertEqual(len(self.segmented_df), 100, "User profile count is not 100")
        self.assertEqual(self.segmented_df.index.nunique(), 100, "User IDs are not unique in user profile")

    # ── Test 5: Valid Cluster IDs & Labels ────────────────────────────────────
    def test_valid_cluster_ids_and_labels(self):
        """Cluster IDs must be integers in [0, 1, 2] with non-empty string labels."""
        valid_ids = {0, 1, 2}
        assigned_ids = set(self.segmented_df["Cluster_ID"].unique())
        self.assertEqual(assigned_ids, valid_ids, f"Unexpected cluster IDs: {assigned_ids}")
        self.assertEqual(self.segmented_df["Cluster_Label"].nunique(), 3, "Did not get 3 unique cluster labels")
        for label in self.segmented_df["Cluster_Label"]:
            self.assertIsInstance(label, str)
            self.assertGreater(len(label), 0)

    # ── Test 6: Output Files Exist on Disk ─────────────────────────────────────
    def test_output_files_exist(self):
        """Verify daily_wellness_scores.csv, user_lifestyle_profiles.csv, and figures exist."""
        self.assertTrue(os.path.exists(self.daily_csv), f"Missing {self.daily_csv}")
        self.assertTrue(os.path.exists(self.user_csv), f"Missing {self.user_csv}")

        daily = pd.read_csv(self.daily_csv)
        self.assertEqual(daily.shape[0], 36500)
        self.assertIn("Wellness_Score", daily.columns)

        users = pd.read_csv(self.user_csv)
        self.assertEqual(users.shape[0], 100)
        self.assertIn("Cluster_ID", users.columns)
        self.assertIn("Cluster_Label", users.columns)

        # Check 7 visualization files
        expected_figs = [
            "01_silhouette_vs_k.png",
            "02_cluster_size_distribution.png",
            "03_average_activity_by_cluster.png",
            "04_average_sleep_by_cluster.png",
            "05_average_stress_by_cluster.png",
            "06_average_wellness_by_cluster.png",
            "07_pca_2d_user_clusters.png",
        ]
        for fig_name in expected_figs:
            fig_path = os.path.join(self.fig_dir, fig_name)
            self.assertTrue(os.path.exists(fig_path), f"Missing figure {fig_name}")
            self.assertGreater(os.path.getsize(fig_path), 1000, f"Figure {fig_name} is empty")


if __name__ == "__main__":
    unittest.main()
