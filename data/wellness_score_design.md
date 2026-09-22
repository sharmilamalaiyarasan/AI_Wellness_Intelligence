# Wellness Score Design Specification

**Platform:** AI-Powered Fitness & Lifestyle Wellness Intelligence Platform  
**System Module:** PS15 – Fitness & Lifestyle Wellness Analytics  
**Document Path:** `data/wellness_score_design.md`  
**Dataset Version:** `data/processed/fitness_health_cleaned.csv`  

---

## 1. Executive Summary & Objective

The primary objective of the **Daily Wellness Score** is to provide an objective, holistic, 0–100 scale index summarizing an individual's daily lifestyle balance. It synthesizes five distinct dimensions—**Sleep, Physical Activity, Hydration, Stress Management, and Resting Heart Rate**—derived directly from tracked wearable/lifestyle data.

### Scope & Ethical Framing
> [!IMPORTANT]
> **Non-Diagnostic Framing:** This Wellness Score is an analytical and educational index for lifestyle pacing, self-reflection, and healthy habit reinforcement. It is **NOT** a medical diagnostic instrument, clinical triage tool, or health prognosis. No medical treatment decisions should be based on this score.

---

## 2. Dataset Distribution & Empirical Baselines

Before fixing thresholds, the empirical distribution of the 36,500 continuous person-day records (100 distinct users, 365 days each) in `data/processed/fitness_health_cleaned.csv` was analyzed:

| Feature | Min | 25th %ile | Median | Mean | 75th %ile | Max | Std Dev |
|---|---|---|---|---|---|---|---|
| **`Hours_Slept`** | 4.00 h | 5.38 h | 6.49 h | 6.49 h | 7.58 h | 9.00 h | 1.36 h |
| **`Active_Minutes`** | 30.0 min | 53.0 min | 74.5 min | 74.4 min | 96.0 min | 119.0 min | 25.3 min |
| **`Steps_Taken`** | 2,000 | 6,523 | 11,042 | 11,035 | 15,562 | 19,999 | 5,203 |
| **`Calories_Burned`** | 1,500 kcal | 2,119 kcal | 2,748 kcal | 2,748 kcal | 3,376 kcal | 3,999 kcal | 724 kcal |
| **`Water_Intake (Liters)`** | 1.50 L | 2.23 L | 2.75 L | 2.74 L | 3.25 L | 4.00 L | 0.67 L |
| **`Heart_Rate (bpm)`** | 60.0 bpm | 70.0 bpm | 80.0 bpm | 79.6 bpm | 89.0 bpm | 99.0 bpm | 11.2 bpm |
| **`Stress_Level (1–10)`** | 1.0 | 3.0 | 6.0 | 5.5 | 8.0 | 10.0 | 2.88 |

---

## 3. Candidate Dimensions & Component Formulation

Each dimension is normalized onto a continuous $[0, 100]$ interval using continuous piece-wise linear transforms to prevent step-function boundary artifacts.

```
+-----------------------------------------------------------------------------+
|                               WELLNESS SCORE                                |
|                                   (0-100)                                   |
+-----------------------------------------------------------------------------+
        |                  |                 |                 |            |
     [25%]              [25%]             [20%]             [15%]        [15%]
        v                  v                 v                 v            v
  +-----------+     +------------+     +-----------+     +-----------+ +-----------+
  |   SLEEP   |     |  ACTIVITY  |     |  STRESS   |     | HYDRATION | |HEART RATE |
  |   SCORE   |     |   SCORE    |     |   SCORE   |     |   SCORE   | |   SCORE   |
  +-----------+     +------------+     +-----------+     +-----------+ +-----------+
```

---

### Component 1: Sleep Score ($S_{\text{sleep}}$) — Weight: 25%

- **Clinical / Guideline Rationale:** The National Sleep Foundation and American Academy of Sleep Medicine recommend 7.0–9.0 hours of restorative sleep per night for adults.
- **Empirical Boundary in Dataset:** Sleep hours range between 4.0 and 9.0 hours.
- **Mathematical Definition:**
  $$\text{If } H \ge 7.0: \quad S_{\text{sleep}} = 100.0$$
  $$\text{If } H < 7.0: \quad S_{\text{sleep}} = \text{clip}\left( \frac{H - 4.0}{7.0 - 4.0} \times 100.0, \, 0.0, \, 100.0 \right)$$
- **Score Behavior:**
  - $9.0\text{ h} \rightarrow 100$
  - $7.0\text{ h} \rightarrow 100$
  - $6.0\text{ h} \rightarrow 66.7$
  - $5.0\text{ h} \rightarrow 33.3$
  - $\le 4.0\text{ h} \rightarrow 0.0$

---

### Component 2: Physical Activity Score ($S_{\text{activity}}$) — Weight: 25%

- **Guideline Rationale:** The World Health Organization (WHO) and CDC recommend 150–300 minutes of moderate activity weekly (~30–60 minutes daily) and reaching 7,500–10,000 steps daily.
- **Dual Indicator Approach:** Both volume (`Steps_Taken`) and intensity duration (`Active_Minutes`) contribute equally (50% each) to avoid penalizing individuals engaged in non-step cardiovascular exercises (e.g., cycling, yoga, swimming).
- **Mathematical Definition:**
  $$S_{\text{steps}} = \text{clip}\left( \frac{\text{Steps} - 2000}{10000 - 2000} \times 100.0, \, 0.0, \, 100.0 \right)$$
  $$S_{\text{mins}} = \text{clip}\left( \frac{\text{Active\_Minutes} - 30}{60 - 30} \times 100.0, \, 0.0, \, 100.0 \right)$$
  $$S_{\text{activity}} = 0.5 \times S_{\text{steps}} + 0.5 \times S_{\text{mins}}$$
- **Score Behavior:**
  - $\ge 10,000$ steps and $\ge 60$ active mins $\rightarrow 100.0$
  - 6,000 steps and 45 active mins $\rightarrow 50.0 + 25.0 = 75.0$
  - $\le 2,000$ steps and $\le 30$ active mins $\rightarrow 0.0$

---

### Component 3: Stress Management Score ($S_{\text{stress}}$) — Weight: 20%

- **Guideline Rationale:** Chronic psychological and physiological stress impacts autonomic nervous balance, immune function, and recovery.
- **Dataset Scale:** Self-reported or biosensor-assessed rating on a scale of 1 (low stress, high relaxation) to 10 (maximum acute stress).
- **Mathematical Definition (Direct Inverse Scale):**
  $$S_{\text{stress}} = \text{clip}\left( \frac{10 - \text{Stress}}{10 - 1} \times 100.0, \, 0.0, \, 100.0 \right)$$
- **Score Behavior:**
  - Stress Level 1 (Optimal calm) $\rightarrow 100.0$
  - Stress Level 3 $\rightarrow 77.8$
  - Stress Level 5.5 (Dataset mean) $\rightarrow 50.0$
  - Stress Level 8 $\rightarrow 22.2$
  - Stress Level 10 (Severe stress) $\rightarrow 0.0$

---

### Component 4: Hydration Score ($S_{\text{hydration}}$) — Weight: 15%

- **Guideline Rationale:** General wellness guidelines (National Academies of Sciences, Engineering, and Medicine) recommend approximately 2.7 to 3.7 liters of daily fluid intake for adults depending on climate and activity.
- **Empirical Boundary in Dataset:** Values range from 1.5 to 4.0 liters daily.
- **Mathematical Definition:**
  $$\text{If } W \ge 3.0: \quad S_{\text{hydration}} = 100.0$$
  $$\text{If } W < 3.0: \quad S_{\text{hydration}} = \text{clip}\left( \frac{W - 1.5}{3.0 - 1.5} \times 100.0, \, 0.0, \, 100.0 \right)$$
- **Score Behavior:**
  - $\ge 3.0\text{ L} \rightarrow 100.0$
  - $2.5\text{ L} \rightarrow 66.7$
  - $2.0\text{ L} \rightarrow 33.3$
  - $\le 1.5\text{ L} \rightarrow 0.0$

---

### Component 5: Resting Heart Rate Score ($S_{\text{heart\_rate}}$) — Weight: 15%

- **Guideline Rationale:** American Heart Association (AHA) defines standard adult resting heart rate between 60 and 100 bpm. Lower resting heart rates (60–75 bpm) reflect higher cardiorespiratory fitness, parasympathetic tone, and stroke volume efficiency.
- **Empirical Boundary in Dataset:** Ranges between 60.0 and 99.0 bpm.
- **Mathematical Definition:**
  $$\text{If } HR \le 65: \quad S_{\text{heart\_rate}} = 100.0$$
  $$\text{If } HR > 65: \quad S_{\text{heart\_rate}} = \text{clip}\left( 100.0 - \left(\frac{HR - 65}{95 - 65}\right) \times 80.0, \, 0.0, \, 100.0 \right)$$
- **Score Behavior:**
  - $60–65\text{ bpm} \rightarrow 100.0$
  - $75\text{ bpm} \rightarrow 73.3$
  - $85\text{ bpm} \rightarrow 46.7$
  - $\ge 95\text{ bpm} \rightarrow \le 20.0$

---

## 4. Master Composite Formula

The master daily **Wellness Score** ($W$) is the linear convex combination of the five normalized components:

$$W = 0.25 \times S_{\text{sleep}} + 0.25 \times S_{\text{activity}} + 0.20 \times S_{\text{stress}} + 0.15 \times S_{\text{hydration}} + 0.15 \times S_{\text{heart\_rate}}$$

Bounded strictly in $[0.0, 100.0]$.

### Qualitative Score Bands
- **85.0 – 100.0:** 🌟 *Optimal Wellness* (Exemplary balance of recovery, activity, and vital signs)
- **70.0 – 84.9:** 🟢 *Balanced Lifestyle* (Healthy habits with minor room for optimization)
- **50.0 – 69.9:** 🟡 *Moderate Lifestyle* (Fair habits; noticeable recovery or stress deficits)
- **0.0 – 49.9:** 🔴 *Needs Attention* (Multiple lifestyle stressors: low sleep, low activity, or elevated stress)

---

## 5. Prevention of Target Leakage

In subsequent ML modeling (Phase 4):
1. The Wellness Score $W$ is a **derived target variable** ($y$).
2. The exact sub-component scores ($S_{\text{sleep}}, S_{\text{activity}}$, etc.) MUST NOT be used as input features ($X$) in the regression model, as that would constitute direct trivial leakage ($X \rightarrow y$).
3. The predictive model must take solely the **raw lifestyle signals** (or time-lagged historical moving averages) to infer future wellness trajectory.
4. The formulation is completely deterministic, closed-form, and reproducible across any wearable time-series dataset.

---

## 6. Limitations & Technical Boundaries

1. **Synthetic/Simulated Sensor Bounds:** The Kaggle dataset exhibits uniform-like truncations (e.g. sleep strictly between 4.0h and 9.0h; steps between 2,000 and 19,999). Extreme medical outliers (e.g. sleep apnea, severe tachycardia) are absent from the underlying dataset.
2. **Lack of Dietary Macronutrient Data:** Caloric expenditure is available, but dietary caloric intake or food composition is absent.
3. **No Direct Sleep Architecture:** Deep sleep, REM, and sleep interruptions are not tracked separately; sleep quality is indexed via total duration.
