"""
User Behaviour Segmentation Module
==================================
Performs unsupervised lifestyle clustering on 365-day aggregated user profiles (100 users).
Uses StandardScaler, KMeans clustering, silhouette scoring, and PCA 2D projection.
Produces explainable behavioral personas based on empirical cluster centers.
"""

from typing import Dict, Any, Tuple, List, Optional
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from analytics.lifestyle_insights import get_user_level_wellness_summaries

# Default segmentation feature set
SEGMENTATION_FEATURES = [
    "Avg_Steps",
    "Avg_Active_Mins",
    "Avg_Sleep_Hours",
    "Avg_Calories",
    "Avg_Water_Liters",
    "Avg_Heart_Rate",
    "Avg_Stress",
    "Workout_Frequency",
    "Avg_Wellness_Score",
]


def evaluate_cluster_range(
    user_df: pd.DataFrame,
    features: List[str] = SEGMENTATION_FEATURES,
    k_range: range = range(2, 7),
    random_state: int = 42,
) -> Dict[int, Dict[str, Any]]:
    """
    Evaluate multiple cluster counts (K = 2 to 6) using Inertia and Silhouette Score.
    Returns dictionary mapping K -> {'silhouette': float, 'inertia': float, 'cluster_sizes': list}.
    """
    X = user_df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels))
        inertia = float(km.inertia_)
        counts = [int(c) for c in np.bincount(labels)]
        results[k] = {
            "k": k,
            "silhouette": round(sil, 4),
            "inertia": round(inertia, 2),
            "cluster_sizes": counts,
        }
    return results


def perform_segmentation(
    user_df: pd.DataFrame,
    n_clusters: int = 3,
    features: List[str] = SEGMENTATION_FEATURES,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, KMeans, StandardScaler, Dict[int, str]]:
    """
    Perform KMeans segmentation on aggregated user data.
    Generates meaningful descriptive labels based on empirical cluster centroids.
    Returns: (segmented_df, fitted_kmeans, fitted_scaler, label_mapping)
    """
    out_df = user_df.copy()
    X = out_df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    out_df["Cluster_ID"] = labels

    # Compute cluster centroid profiles
    profile = out_df.groupby("Cluster_ID")[features].mean()

    # Assign descriptive labels based on observed centroids:
    # Cluster with highest Wellness & highest Sleep -> Balanced Mover / Optimal Recovery
    # Cluster with highest Active Minutes & lowest Sleep -> Active Mover / Sleep Deficit
    # Cluster with highest Workout Frequency & highest Stress -> Frequent Exerciser / High Stress
    label_mapping = {}
    best_wellness_cid = int(profile["Avg_Wellness_Score"].idxmax())
    highest_active_lowest_sleep_cid = int(profile["Avg_Active_Mins"].idxmax())
    
    for cid in range(n_clusters):
        if cid == best_wellness_cid:
            label_mapping[cid] = "Balanced Mover / Optimal Recovery"
        elif cid == highest_active_lowest_sleep_cid:
            label_mapping[cid] = "Active Mover / Sleep Deficit"
        else:
            label_mapping[cid] = "Frequent Exerciser / High Stress"

    out_df["Cluster_Label"] = out_df["Cluster_ID"].map(label_mapping)

    # 2D PCA projection coordinates for visualization
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)
    out_df["PCA1"] = pca_coords[:, 0].round(3)
    out_df["PCA2"] = pca_coords[:, 1].round(3)

    return out_df, kmeans, scaler, label_mapping


def generate_cluster_plots(
    segmented_user_df: pd.DataFrame,
    eval_results: Dict[int, Dict[str, Any]],
    output_dir: str,
) -> List[str]:
    """
    Generate the 7 required static cluster visualization figures with clean styling:
    1. Silhouette score vs number of clusters
    2. Cluster size distribution
    3. Average activity (steps) by cluster
    4. Average sleep by cluster
    5. Average stress by cluster
    6. Average wellness score by cluster
    7. 2D PCA visualization of user clusters
    Saves PNG figures to output_dir and returns list of created filepaths.
    """
    os.makedirs(output_dir, exist_ok=True)
    created_files = []
    sns.set_theme(style="whitegrid")
    palette = ["#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]

    # ── 1. Silhouette Score vs K ──
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ks = list(eval_results.keys())
    sils = [eval_results[k]["silhouette"] for k in ks]
    ax.plot(ks, sils, marker="o", color="#4F46E5", linewidth=2.5, markersize=8)
    for k, s in zip(ks, sils):
        ax.annotate(f"{s:.4f}", (k, s + 0.0015), ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Silhouette Score vs. Number of Clusters (K)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Clusters (K)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Silhouette Score", fontsize=10, fontweight="bold")
    ax.set_xticks(ks)
    p1 = os.path.join(output_dir, "01_silhouette_vs_k.png")
    fig.tight_layout()
    fig.savefig(p1)
    plt.close(fig)
    created_files.append(p1)

    # ── 2. Cluster Size Distribution ──
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    counts = segmented_user_df["Cluster_Label"].value_counts()
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=12, ha="right", fontsize=9)
    bars = ax.bar(range(len(counts)), counts.values, color=palette[:len(counts)], width=0.55, edgecolor="none")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{int(h)} Users", ha="center", fontweight="bold", fontsize=10)
    ax.set_title("Cluster Size Distribution (N = 100 Users)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Users", fontsize=10, fontweight="bold")
    p2 = os.path.join(output_dir, "02_cluster_size_distribution.png")
    fig.tight_layout()
    fig.savefig(p2)
    plt.close(fig)
    created_files.append(p2)

    # ── 3. Average Activity (Steps) by Cluster ──
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    agg_act = segmented_user_df.groupby("Cluster_Label")["Avg_Steps"].mean()
    ax.set_xticks(range(len(agg_act)))
    ax.set_xticklabels(agg_act.index, rotation=12, ha="right", fontsize=9)
    bars = ax.bar(range(len(agg_act)), agg_act.values, color="#3B82F6", width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 50, f"{h:,.0f} steps", ha="center", fontweight="bold", fontsize=9)
    ax.set_title("Average Daily Steps by Behaviour Cluster", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Average Steps", fontsize=10, fontweight="bold")
    p3 = os.path.join(output_dir, "03_average_activity_by_cluster.png")
    fig.tight_layout()
    fig.savefig(p3)
    plt.close(fig)
    created_files.append(p3)

    # ── 4. Average Sleep by Cluster ──
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    agg_sleep = segmented_user_df.groupby("Cluster_Label")["Avg_Sleep_Hours"].mean()
    ax.set_xticks(range(len(agg_sleep)))
    ax.set_xticklabels(agg_sleep.index, rotation=12, ha="right", fontsize=9)
    bars = ax.bar(range(len(agg_sleep)), agg_sleep.values, color="#10B981", width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{h:.2f} hrs", ha="center", fontweight="bold", fontsize=9)
    ax.set_title("Average Sleep Duration by Behaviour Cluster", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Sleep Duration (Hours)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 8.5)
    p4 = os.path.join(output_dir, "04_average_sleep_by_cluster.png")
    fig.tight_layout()
    fig.savefig(p4)
    plt.close(fig)
    created_files.append(p4)

    # ── 5. Average Stress by Cluster ──
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    agg_stress = segmented_user_df.groupby("Cluster_Label")["Avg_Stress"].mean()
    ax.set_xticks(range(len(agg_stress)))
    ax.set_xticklabels(agg_stress.index, rotation=12, ha="right", fontsize=9)
    bars = ax.bar(range(len(agg_stress)), agg_stress.values, color="#EF4444", width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.08, f"{h:.2f} / 10", ha="center", fontweight="bold", fontsize=9)
    ax.set_title("Average Perceived Stress Rating by Behaviour Cluster", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Stress Level (1–10)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 7.5)
    p5 = os.path.join(output_dir, "05_average_stress_by_cluster.png")
    fig.tight_layout()
    fig.savefig(p5)
    plt.close(fig)
    created_files.append(p5)

    # ── 6. Average Wellness Score by Cluster ──
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    agg_well = segmented_user_df.groupby("Cluster_Label")["Avg_Wellness_Score"].mean()
    ax.set_xticks(range(len(agg_well)))
    ax.set_xticklabels(agg_well.index, rotation=12, ha="right", fontsize=9)
    bars = ax.bar(range(len(agg_well)), agg_well.values, color="#8B5CF6", width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f} / 100", ha="center", fontweight="bold", fontsize=9)
    ax.set_title("Average Composite Wellness Score by Behaviour Cluster", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Wellness Score (0–100)", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 85)
    p6 = os.path.join(output_dir, "06_average_wellness_by_cluster.png")
    fig.tight_layout()
    fig.savefig(p6)
    plt.close(fig)
    created_files.append(p6)

    # ── 7. 2D PCA Visualization ──
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    sns.scatterplot(
        data=segmented_user_df,
        x="PCA1",
        y="PCA2",
        hue="Cluster_Label",
        palette=palette[:segmented_user_df["Cluster_ID"].nunique()],
        s=90,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.8,
        ax=ax,
    )
    ax.set_title("2D Principal Component Projection (PCA) of User Clusters", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Principal Component 1 (PCA 1)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Principal Component 2 (PCA 2)", fontsize=10, fontweight="bold")
    ax.legend(title="Lifestyle Persona", loc="upper right", frameon=True, fontsize=8.5)
    p7 = os.path.join(output_dir, "07_pca_2d_user_clusters.png")
    fig.tight_layout()
    fig.savefig(p7)
    plt.close(fig)
    created_files.append(p7)

    return created_files
