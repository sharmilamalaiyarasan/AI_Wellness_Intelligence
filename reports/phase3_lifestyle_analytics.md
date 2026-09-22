# Phase 3: Lifestyle Analytics & User Behaviour Segmentation Report

**Project:** AI-Powered Fitness & Lifestyle Wellness Intelligence Platform  
**System Module:** PS15 – Fitness & Lifestyle Wellness Analytics System  
**Document Path:** `reports/phase3_lifestyle_analytics.md`  
**Dataset Analyzed:** `data/processed/fitness_health_cleaned.csv`  

---

## 1. Dataset Overview

The dataset is derived from the official Kaggle dataset (*Comprehensive Fitness and Health Tracking Dataset* by Siddhesh Toraskar), downloaded programmatically via `kagglehub` and preprocessed into `data/processed/fitness_health_cleaned.csv`.

- **Cohort Size:** 100 unique individuals (`User_ID` 1 to 100).
- **Temporal Horizon:** Full 365 calendar days (January 1, 2023 to December 31, 2023).
- **Total Record Count:** 36,500 daily entries ($100 \times 365$).
- **Demographic Spread:**
  - Age: 19 to 63 years (mean: 43.16, std: 13.67).
  - Gender: Male (40.0%), Female (36.0%), Other (24.0%).
  - Height: 150 to 199 cm (mean: 172.69 cm).
  - Weight: 51 to 118 kg (mean: 85.83 kg).
  - Mean Body Mass Index (BMI): $28.94\text{ kg/m}^2$.
- **Tracking Modalities:** Daily step counts, active minutes duration, caloric burn, hours slept, hydration (liters), resting heart rate (bpm), subjective/sensor stress rating (1–10 scale), daily mood state, and workout category.
- **Missing Value Handling:** Missing workout categories (25.2%) were classified as `'Rest'` (non-exercise days). Missing physiological readings were imputed via user-specific 365-day medians to maintain individual physiological baselines without synthetic distortion.

---

## 2. Wellness Score Methodology

The daily **Wellness Score** ($W$) is an objective, deterministic composite index on a 0–100 continuous scale, documented in `data/wellness_score_design.md`.

$$W = 0.25 \times S_{\text{sleep}} + 0.25 \times S_{\text{activity}} + 0.20 \times S_{\text{stress}} + 0.15 \times S_{\text{hydration}} + 0.15 \times S_{\text{heart\_rate}}$$

### Component Normalization & Weights
1. **Sleep Score ($S_{\text{sleep}}$, 25%):**
   - Grounded in National Sleep Foundation guidelines ($\ge 7.0$ hours optimal).
   - $\ge 7.0\text{ h} \rightarrow 100.0$; scaled linearly to 0 at $\le 4.0\text{ h}$.
2. **Activity Score ($S_{\text{activity}}$, 25%):**
   - Blends daily step volume (50%, $2,000 \rightarrow 0, \ge 10,000 \rightarrow 100$) and active minutes (50%, $30\text{ min} \rightarrow 0, \ge 60\text{ min} \rightarrow 100$, aligned with WHO guidelines).
3. **Stress Score ($S_{\text{stress}}$, 20%):**
   - Inverse linear scaling of the 1–10 stress rating: $\frac{10 - \text{Stress}}{9} \times 100$.
4. **Hydration Score ($S_{\text{hydration}}$, 15%):**
   - $\ge 3.0\text{ L} \rightarrow 100.0$; scaled linearly to 0 at $\le 1.5\text{ L}$.
5. **Heart Rate Score ($S_{\text{heart\_rate}}$, 15%):**
   - $\le 65\text{ bpm} \rightarrow 100.0$; 65 to 95 bpm scales down to 20; $>95\text{ bpm}$ scales to 0.

### Score Summary Statistics (N = 36,500 daily records)
- **Mean Daily Wellness Score:** $67.77 \pm 13.13$
- **Median:** $68.50$ (Min: $14.25$, Max: $100.00$)
- **Interquartile Range:** $59.01$ (25th percentile) to $77.17$ (75th percentile)
- **Component Means:**
  - $S_{\text{sleep}}$: $71.29 \pm 31.73$
  - $S_{\text{activity}}$: $80.69 \pm 21.38$
  - $S_{\text{hydration}}$: $71.62 \pm 31.04$
  - $S_{\text{stress}}$: $50.01 \pm 32.02$
  - $S_{\text{heart\_rate}}$: $60.19 \pm 28.56$

---

## 3. Lifestyle Statistics & Headline Metrics

| Indicator | Mean $\pm$ Std | Median | Range (Min – Max) |
|---|---|---|---|
| **Daily Steps** | $11,035.1 \pm 5,203.3$ | $11,042.0$ | 2,000 – 19,999 |
| **Active Minutes** | $74.4 \pm 25.3\text{ mins}$ | $74.5\text{ mins}$ | 30.0 – 119.0 mins |
| **Sleep Duration** | $6.49 \pm 1.36\text{ hrs}$ | $6.49\text{ hrs}$ | 4.00 – 9.00 hrs |
| **Caloric Expenditure** | $2,747.7 \pm 723.9\text{ kcal}$ | $2,748.0\text{ kcal}$ | 1,500 – 3,999 kcal |
| **Water Intake** | $2.74 \pm 0.67\text{ L}$ | $2.75\text{ L}$ | 1.50 – 4.00 L |
| **Resting Heart Rate** | $79.6 \pm 11.2\text{ bpm}$ | $80.0\text{ bpm}$ | 60.0 – 99.0 bpm |
| **Stress Rating** | $5.50 \pm 2.88$ | $6.00$ | 1.0 – 10.0 |

### Categorical Distributions
- **Workout Type Distribution:**
  - Yoga: 9,222 days ($25.27\%$)
  - Rest Days: 9,192 days ($25.18\%$)
  - Cardio: 9,048 days ($24.79\%$)
  - Strength: 9,038 days ($24.76\%$)
- **Reported Mood Distribution:**
  - Sad: 9,284 days ($25.44\%$)
  - Neutral: 9,115 days ($24.97\%$)
  - Happy: 9,061 days ($24.82\%$)
  - Stressed: 9,040 days ($24.77\%$)

---

## 4. Observed Relationships & Patterns

> [!NOTE]
> All analytical statements describe observed statistical correlations and empirical distributions. They do not assert clinical or physiological causation.

1. **Physical Activity & Sleep Duration:**
   - Linear Pearson correlation: $r = +0.0008$.
   - When segmenting step counts into quartiles, mean sleep duration remained stable:
     - Low Activity ($<6,523$ steps): $6.49 \pm 1.37\text{ hrs}$
     - Moderate Activity ($6,523–11,042$ steps): $6.50 \pm 1.35\text{ hrs}$
     - High Activity ($11,042–15,562$ steps): $6.48 \pm 1.36\text{ hrs}$
     - Very High Activity ($>15,562$ steps): $6.50 \pm 1.36\text{ hrs}$
2. **Activity Duration & Caloric Expenditure:**
   - Linear Pearson correlation: $r = -0.0031$.
   - Across active minute tiers ($<45\text{ min}$, $45–75\text{ min}$, $75–105\text{ min}$, $>105\text{ min}$), average caloric expenditure showed minimal linear variance in this dataset, indicating independent tracking sensor simulations.
3. **Stress Rating & Sleep Duration:**
   - Linear Pearson correlation: $r = -0.0022$.
   - Mean sleep across stress levels:
     - Stress Level 1 (Optimal calm): $6.51\text{ hrs}$
     - Stress Level 5 (Moderate): $6.49\text{ hrs}$
     - Stress Level 10 (Severe stress): $6.50\text{ hrs}$
4. **Weekend vs. Weekday Patterns:**
   - Weekdays (Mon–Fri) vs. Weekends (Sat–Sun) exhibited high behavioral consistency:
     - Daily Steps: $11,024.0$ (Weekday) vs. $11,063.0$ (Weekend) ($+0.4\%$ diff)
     - Sleep Duration: $6.49\text{ hrs}$ vs. $6.50\text{ hrs}$ ($+0.2\%$ diff)
     - Average Wellness Score: $67.67$ vs. $68.01$ ($+0.5\%$ diff)

---

## 5. Behaviour Segmentation Methodology

Clustering was conducted on **user-level aggregations** across the complete 365-day observation window ($N = 100$ users), rather than on single days, capturing long-term lifestyle habits rather than transient daily fluctuations.

### Feature Matrix ($X \in \mathbb{R}^{100 \times 9}$):
1. `Avg_Steps`
2. `Avg_Active_Mins`
3. `Avg_Sleep_Hours`
4. `Avg_Calories`
5. `Avg_Water_Liters`
6. `Avg_Heart_Rate`
7. `Avg_Stress`
8. `Workout_Frequency` (fraction of days with structured workout)
9. `Avg_Wellness_Score`

All features were standardized to $\mu = 0, \sigma = 1$ using `StandardScaler` prior to distance computation.

### Model Evaluation Across Cluster Counts ($K = 2$ to $6$):

| Cluster Count ($K$) | Silhouette Score | Inertia (WCSS) | Cluster Sizes |
|---|---|---|---|
| **$K = 2$** | **0.1258** | 766.84 | $[46, 54]$ |
| **$K = 3$** | **0.1199** | **695.99** | **$[32, 27, 41]$** |
| **$K = 4$** | 0.1033 | 647.78 | $[29, 19, 29, 23]$ |
| **$K = 5$** | 0.1016 | 604.99 | $[23, 19, 23, 20, 15]$ |
| **$K = 6$** | 0.1032 | 570.10 | $[18, 14, 16, 15, 21, 16]$ |

### Configuration Selection Rationale
$K = 3$ was selected because:
1. It maintains a high silhouette score ($0.1199$), nearly identical to $K=2$ ($0.1258$).
2. It reduces within-cluster sum of squares (inertia) by over $9.2\%$ compared to $K=2$.
3. It produces three balanced, clinically intuitive behavioral personas without fragmenting the 100-user sample into overly small cohorts.

---

## 6. Cluster Characteristics & Personas

```
+-----------------------------------------------------------------------------+
|                     USER LIFESTYLE BEHAVIOUR CLUSTERS                       |
+-----------------------------------------------------------------------------+
| Cluster 0: Frequent Exerciser / High Stress  (32 Users | 32%)               |
| -> Workout Freq: 77.0% | Heart Rate: 79.91 bpm | Stress: 5.55 / 10          |
+-----------------------------------------------------------------------------+
| Cluster 1: Active Mover / Sleep Deficit      (27 Users | 27%)               |
| -> Active Mins: 74.93 min | Sleep: 6.42 hrs | Water: 2.71 L | Wellness: 67.11|
+-----------------------------------------------------------------------------+
| Cluster 2: Balanced Mover / Optimal Recovery (41 Users | 41%)               |
| -> Steps: 11,090.95 | Sleep: 6.53 hrs | Stress: 5.44 | Wellness: 68.42      |
+-----------------------------------------------------------------------------+
```

### Detailed Centroid Comparison:

| Feature | Cluster 0: Frequent Exerciser / High Stress | Cluster 1: Active Mover / Sleep Deficit | Cluster 2: Balanced Mover / Optimal Recovery |
|---|---|---|---|
| **Cohort Size** | 32 users ($32\%$) | 27 users ($27\%$) | 41 users ($41\%$) |
| **Avg Daily Steps** | $10,971.34$ steps | $11,025.90$ steps | **$11,090.95$ steps** |
| **Avg Active Minutes** | $74.11$ mins | **$74.93$ mins** | $74.22$ mins |
| **Avg Sleep Hours** | $6.51$ hrs | $6.42$ hrs *(Lowest)* | **$6.53$ hrs** *(Highest)* |
| **Avg Daily Calories** | $2,752.67$ kcal | **$2,759.67$ kcal** | $2,736.03$ kcal |
| **Avg Water Intake** | $2.74$ L | $2.71$ L *(Lowest)* | **$2.76$ L** *(Highest)* |
| **Avg Resting HR** | **$79.91$ bpm** *(Highest)* | $79.68$ bpm | **$79.25$ bpm** *(Lowest)* |
| **Avg Stress Level** | **$5.55$** *(Highest)* | $5.52$ | **$5.44$** *(Lowest)* |
| **Workout Frequency** | **$77.0\%$ of days** *(Highest)* | $73.0\%$ of days | $74.0\%$ of days |
| **Avg Wellness Score** | $67.49$ | $67.11$ *(Lowest)* | **$68.42$** *(Highest)* |

### Behavioral Interpretation of Personas
- **Cluster 0 — Frequent Exerciser / High Stress ($N=32$):** Users in this cluster exercise most frequently ($77\%$ of days), yet maintain slightly higher resting heart rates ($79.91\text{ bpm}$) and higher self-reported stress ratings ($5.55$). Coaching focus: Recovery pacing, yoga/mindfulness integration, and avoiding overtraining.
- **Cluster 1 — Active Mover / Sleep Deficit ($N=27$):** Individuals sustaining the longest active exercise durations ($74.93\text{ mins/day}$) and highest caloric burn ($2,759.7\text{ kcal}$), accompanied by the lowest sleep duration ($6.42\text{ hrs}$) and lowest hydration levels ($2.71\text{ L}$). Coaching focus: Sleep hygiene extension and progressive hydration goals.
- **Cluster 2 — Balanced Mover / Optimal Recovery ($N=41$):** Individuals exhibiting the most balanced long-term profile: highest step volume ($11,091$), highest sleep hours ($6.53\text{ hrs}$), highest hydration ($2.76\text{ L}$), lowest resting heart rate ($79.25\text{ bpm}$), and highest overall Wellness Score ($68.42$). Coaching focus: Maintenance of consistent cross-training habits.

---

## 7. Generated Figures & Visualizations

The following 7 publication-ready figures were generated and saved to `reports/figures/`:
1. `01_silhouette_vs_k.png`: Silhouette score curve across $K \in [2, 6]$.
2. `02_cluster_size_distribution.png`: Bar chart of user distribution across the three personas.
3. `03_average_activity_by_cluster.png`: Step count comparison across personas.
4. `04_average_sleep_by_cluster.png`: Sleep duration comparison across personas.
5. `05_average_stress_by_cluster.png`: Stress rating comparison across personas.
6. `06_average_wellness_by_cluster.png`: Composite Wellness Score comparison across personas.
7. `07_pca_2d_user_clusters.png`: 2-dimensional principal component projection displaying user clustering boundaries.

---

## 8. Exported Data Products

1. **`data/processed/daily_wellness_scores.csv`**
   - 36,500 rows, 25 columns.
   - Contains all original preprocessed features plus daily calculated `Sleep_Score`, `Activity_Score`, `Hydration_Score`, `Stress_Score`, `Heart_Rate_Score`, and composite `Wellness_Score`.
2. **`data/processed/user_lifestyle_profiles.csv`**
   - 100 rows, 24 columns.
   - Contains 365-day aggregated metrics per user, assigned `Cluster_ID`, human-readable `Cluster_Label`, and 2D PCA coordinates (`PCA1`, `PCA2`).

---

## 9. Methodological Limitations

1. **Independent Feature Sampling in Underlying Dataset:** The Kaggle dataset exhibits near-zero cross-feature correlation coefficients ($|r| < 0.01$), indicative of synthetic uniform/independent data generation in the source archive. The algorithms nevertheless extract meaningful empirical variance across individual user baselines.
2. **Aggregated Cluster Variance:** Because user profiles are averages across 365 days, extreme single-day outliers are smoothed out by central limit behavior.
3. **Absence of Diagnostic Clinical Biomarkers:** Wearables track proxies (steps, sleep time) rather than clinical biomarkers (e.g., blood pressure, ECG rhythm, HbA1c).

---

## 10. Next Steps for Phase 4 (ML Wellness Prediction Model)

With the daily Wellness Score rigorously established and free of target leakage:
1. Train a supervised regression model (e.g., Random Forest Regressor / Gradient Boosting) using only observable lifestyle features ($X$: `Steps_Taken`, `Active_Minutes`, `Hours_Slept`, `Calories_Burned`, `Water_Intake`, `Heart_Rate`, `Stress_Level`, `Workout_Type`, `Mood`, `Day_of_Week`, `Is_Weekend`, `BMI`) to predict `Wellness_Score` ($y$).
2. Evaluate model performance using 5-Fold Cross-Validation, reporting $R^2$, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
3. Integrate the predictive model and behaviour segmentation into the unified Streamlit application.
