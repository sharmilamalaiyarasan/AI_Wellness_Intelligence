"""
Phase 3 Analysis Pipeline Runner
================================
Calculates daily wellness scores, aggregates user-level profiles,
evaluates cluster configurations, runs KMeans segmentation,
exports output CSVs, and renders visual analytics figures.
"""

import os
import pandas as pd
from analytics.wellness_score import calculate_wellness_score
from analytics.lifestyle_insights import get_user_level_wellness_summaries
from analytics.segmentation import evaluate_cluster_range, perform_segmentation, generate_cluster_plots

def run():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "processed")
    reports_dir = os.path.join(base_dir, "reports")
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Load cleaned data
    cleaned_csv = os.path.join(data_dir, "fitness_health_cleaned.csv")
    df = pd.read_csv(cleaned_csv)
    print(f"Loaded cleaned dataset from {cleaned_csv}: {df.shape}")

    # 2. Calculate daily wellness scores
    scored_daily = calculate_wellness_score(df)
    daily_out = os.path.join(data_dir, "daily_wellness_scores.csv")
    scored_daily.to_csv(daily_out, index=False)
    print(f"Saved daily wellness scores ({scored_daily.shape}) to {daily_out}")

    # 3. User level aggregation
    user_df = get_user_level_wellness_summaries(scored_daily)
    print(f"Aggregated user summaries: {user_df.shape} (100 users)")

    # 4. Evaluate cluster range K=2..6
    eval_results = evaluate_cluster_range(user_df)
    print("\n=== Silhouette & Inertia Evaluation ===")
    for k, res in eval_results.items():
        print(f"K={k}: Silhouette={res['silhouette']:.4f}, Inertia={res['inertia']:.2f}, Sizes={res['cluster_sizes']}")

    # 5. Segment users with K=3
    segmented_users, kmeans, scaler, label_mapping = perform_segmentation(user_df, n_clusters=3)
    print("\nCluster labels assigned:")
    for cid, label in label_mapping.items():
        print(f"  Cluster {cid} -> {label}")

    print("\nCluster distribution:")
    for label, count in segmented_users["Cluster_Label"].value_counts().items():
        print(f"  {label}: {count} users")

    # 6. Save user lifestyle profiles
    user_out = os.path.join(data_dir, "user_lifestyle_profiles.csv")
    segmented_users.to_csv(user_out, index=True)
    print(f"\nSaved user profiles ({segmented_users.shape}) to {user_out}")

    # 7. Generate cluster plots
    plots = generate_cluster_plots(segmented_users, eval_results, figures_dir)
    print(f"\nGenerated {len(plots)} cluster visualization plots in {figures_dir}:")
    for p in plots:
        print(f"  - {os.path.basename(p)}")

    # Also mirror to c:\Users\HP\fitness\AI_Wellness_Intelligence if present
    alt_base = r"C:\Users\HP\fitness\AI_Wellness_Intelligence"
    if os.path.exists(alt_base):
        alt_data = os.path.join(alt_base, "data", "processed")
        alt_fig = os.path.join(alt_base, "reports", "figures")
        os.makedirs(alt_data, exist_ok=True)
        os.makedirs(alt_fig, exist_ok=True)
        scored_daily.to_csv(os.path.join(alt_data, "daily_wellness_scores.csv"), index=False)
        segmented_users.to_csv(os.path.join(alt_data, "user_lifestyle_profiles.csv"), index=True)
        generate_cluster_plots(segmented_users, eval_results, alt_fig)

    print("\nPipeline execution completed successfully!")

if __name__ == "__main__":
    run()
