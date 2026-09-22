import os, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings("ignore")

BASE_DIR   = r"c:\Users\HP\AI-Fitness\AI_Wellness_Intelligence"
DATA_PATH  = os.path.join(BASE_DIR, "data", "processed", "daily_wellness_scores.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
FIG_DIR    = os.path.join(REPORT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

RAW_FEATURES = [
    "Age","Gender","BMI","Steps_Taken","Active_Minutes","Calories_Burned",
    "Hours_Slept","Water_Intake (Liters)","Heart_Rate (bpm)",
    "Stress_Level (1-10)","Is_Weekend"
]
TARGET = "Wellness_Score"

df = pd.read_csv(DATA_PATH)
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"].astype(str))
X = df[RAW_FEATURES].copy()
y = df[TARGET].copy()
mask = X.notna().all(axis=1) & y.notna()
X, y = X[mask], y[mask]
print(f"Data: {len(X)} rows, {X.shape[1]} features")

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

models = {
    "Linear Regression": Pipeline([("s",StandardScaler()),("m",LinearRegression())]),
    "Ridge": Pipeline([("s",StandardScaler()),("m",Ridge(alpha=1.0))]),
    "Lasso": Pipeline([("s",StandardScaler()),("m",Lasso(alpha=0.1,max_iter=5000))]),
    "Random Forest": Pipeline([("s",StandardScaler()),("m",RandomForestRegressor(n_estimators=200,max_depth=12,min_samples_leaf=4,random_state=42,n_jobs=-1))]),
    "Gradient Boosting": Pipeline([("s",StandardScaler()),("m",GradientBoostingRegressor(n_estimators=200,max_depth=5,learning_rate=0.08,subsample=0.85,random_state=42))]),
    "Extra Trees": Pipeline([("s",StandardScaler()),("m",ExtraTreesRegressor(n_estimators=200,max_depth=12,min_samples_leaf=4,random_state=42,n_jobs=-1))]),
}

kf = KFold(n_splits=5,shuffle=True,random_state=42)
results = {}
for name,pipe in models.items():
    print(f"  Training {name}...", end=" ", flush=True)
    pipe.fit(X_train,y_train)
    yp = pipe.predict(X_test)
    mae  = mean_absolute_error(y_test,yp)
    rmse = float(np.sqrt(mean_squared_error(y_test,yp)))
    r2   = r2_score(y_test,yp)
    cv   = cross_val_score(pipe,X_train,y_train,cv=kf,scoring="r2",n_jobs=-1).mean()
    results[name]={"pipe":pipe,"yp":yp,"MAE":round(mae,4),"RMSE":round(rmse,4),"R2":round(r2,4),"CV_R2":round(cv,4)}
    print(f"MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.4f} CV-R2={cv:.4f}")

best_name = max(results,key=lambda n:results[n]["R2"])
best_r2 = results[best_name]["R2"]
print(f"Best: {best_name}  R2={best_r2}")
best_pipe = results[best_name]["pipe"]
y_pred    = results[best_name]["yp"]

# Feature importance
model_obj = best_pipe.named_steps["m"]
if hasattr(model_obj,"feature_importances_"):
    fi = pd.DataFrame({"Feature":RAW_FEATURES,"Importance":model_obj.feature_importances_}).sort_values("Importance",ascending=False)
elif hasattr(model_obj,"coef_"):
    fi = pd.DataFrame({"Feature":RAW_FEATURES,"Importance":abs(model_obj.coef_)}).sort_values("Importance",ascending=False)
else:
    fi = pd.DataFrame()
if not fi.empty:
    print("Feature Importance:")
    print(fi.to_string(index=False))

# Save model
payload = {"pipeline":best_pipe,"feature_names":RAW_FEATURES,"model_name":best_name}
joblib.dump(payload, os.path.join(MODEL_DIR,"wellness_predictor.pkl"))
print("Model saved to models/wellness_predictor.pkl")

# Save metrics json
metrics = {n:{k:v for k,v in r.items() if k not in ("pipe","yp")} for n,r in results.items()}
metrics["best_model"] = best_name
with open(os.path.join(REPORT_DIR,"ml_model_metrics.json"),"w") as f:
    json.dump(metrics,f,indent=2)
print("Metrics saved to reports/ml_model_metrics.json")

# Plots
PALETTE = {"bg":"#0d1117","card":"#161b22","border":"#30363d","a1":"#58a6ff","a2":"#3fb950","a3":"#f78166","a4":"#d2a8ff","text":"#e6edf3","sub":"#8b949e"}

def setup_ax(ax,title="",xlabel="",ylabel=""):
    ax.set_facecolor(PALETTE["card"])
    ax.tick_params(colors=PALETTE["text"],labelsize=9)
    ax.xaxis.label.set_color(PALETTE["text"])
    ax.yaxis.label.set_color(PALETTE["text"])
    for sp in ax.spines.values(): sp.set_edgecolor(PALETTE["border"])
    if title:  ax.set_title(title,color=PALETTE["text"],fontsize=11,pad=8)
    if xlabel: ax.set_xlabel(xlabel,color=PALETTE["sub"])
    if ylabel: ax.set_ylabel(ylabel,color=PALETTE["sub"])

# Plot 1: Model comparison
names  = list(results.keys())
r2s    = [results[n]["R2"] for n in names]
maes   = [results[n]["MAE"] for n in names]
cv_r2s = [results[n]["CV_R2"] for n in names]
clrs   = [PALETTE["a1"],PALETTE["a2"],PALETTE["a3"],PALETTE["a4"],"#ffa657","#79c0ff"]

fig,axes = plt.subplots(1,3,figsize=(16,5))
fig.patch.set_facecolor(PALETTE["bg"])
fig.suptitle("Model Comparison — Wellness Score Prediction",color=PALETTE["text"],fontsize=14)
ax=axes[0]; bars=ax.bar(names,maes,color=clrs[:len(names)],edgecolor=PALETTE["border"])
setup_ax(ax,"MAE (lower is better)","Model","MAE")
ax.bar_label(bars,fmt="%.3f",color=PALETTE["text"],fontsize=8,padding=2)
ax.set_xticklabels(names,rotation=28,ha="right",fontsize=8)
ax=axes[1]; bars=ax.bar(names,r2s,color=clrs[:len(names)],edgecolor=PALETTE["border"])
setup_ax(ax,"R2 Score (higher is better)","Model","R2")
ax.bar_label(bars,fmt="%.4f",color=PALETTE["text"],fontsize=8,padding=2)
ax.set_xticklabels(names,rotation=28,ha="right",fontsize=8)
ax.set_ylim(max(0,min(r2s)-0.05),1.05)
ax=axes[2]; bars=ax.bar(names,cv_r2s,color=clrs[:len(names)],edgecolor=PALETTE["border"])
setup_ax(ax,"5-Fold CV R2 (higher is better)","Model","CV R2")
ax.bar_label(bars,fmt="%.4f",color=PALETTE["text"],fontsize=8,padding=2)
ax.set_xticklabels(names,rotation=28,ha="right",fontsize=8)
ax.set_ylim(max(0,min(cv_r2s)-0.05),1.05)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"phase4_model_comparison.png"),dpi=150,bbox_inches="tight",facecolor=PALETTE["bg"])
plt.close()
print("Plot: model comparison saved")

# Plot 2: Actual vs Predicted
fig,axes=plt.subplots(1,2,figsize=(13,5))
fig.patch.set_facecolor(PALETTE["bg"])
fig.suptitle(f"Actual vs Predicted - {best_name}",color=PALETTE["text"],fontsize=13)
ax=axes[0]; ax.set_facecolor(PALETTE["card"])
ax.scatter(y_test,y_pred,alpha=0.3,s=8,color=PALETTE["a1"],edgecolors="none")
lim=[min(float(y_test.min()),float(y_pred.min()))-2,max(float(y_test.max()),float(y_pred.max()))+2]
ax.plot(lim,lim,"--",color=PALETTE["a3"],linewidth=1.5,label="Perfect fit")
setup_ax(ax,"Actual vs Predicted","Actual Wellness Score","Predicted")
ax.set_xlim(lim);ax.set_ylim(lim)
ax.legend(facecolor=PALETTE["card"],labelcolor=PALETTE["text"],fontsize=8)
ax=axes[1]; ax.set_facecolor(PALETTE["card"])
residuals=y_test.values-y_pred
ax.scatter(y_pred,residuals,alpha=0.3,s=8,color=PALETTE["a2"],edgecolors="none")
ax.axhline(0,color=PALETTE["a3"],linewidth=1.5,linestyle="--")
setup_ax(ax,"Residual Plot","Predicted","Residual (Actual - Predicted)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"phase4_actual_vs_predicted.png"),dpi=150,bbox_inches="tight",facecolor=PALETTE["bg"])
plt.close()
print("Plot: actual vs predicted saved")

# Plot 3: Feature importance
if not fi.empty:
    fig,ax=plt.subplots(figsize=(10,6))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["card"])
    clrs2=[PALETTE["a1"] if i==0 else PALETTE["a2"] if i<3 else PALETTE["sub"] for i in range(len(fi))]
    bars=ax.barh(fi["Feature"].iloc[::-1],fi["Importance"].iloc[::-1],color=clrs2[::-1],edgecolor=PALETTE["border"])
    setup_ax(ax,f"Feature Importance - {best_name}","Importance","Feature")
    ax.bar_label(bars,fmt="%.4f",color=PALETTE["text"],fontsize=8,padding=2)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR,"phase4_feature_importance.png"),dpi=150,bbox_inches="tight",facecolor=PALETTE["bg"])
    plt.close()
    print("Plot: feature importance saved")

# Plot 4: Error distribution
errors=y_test.values-y_pred
fig,axes=plt.subplots(1,2,figsize=(13,5))
fig.patch.set_facecolor(PALETTE["bg"])
fig.suptitle(f"Error Distribution - {best_name}",color=PALETTE["text"],fontsize=13)
ax=axes[0];ax.set_facecolor(PALETTE["card"])
ax.hist(errors,bins=60,color=PALETTE["a1"],edgecolor=PALETTE["bg"],alpha=0.85)
ax.axvline(0,color=PALETTE["a3"],linestyle="--",linewidth=1.5)
setup_ax(ax,"Residual Histogram","Residual","Count")
ax=axes[1];ax.set_facecolor(PALETTE["card"])
sorted_e=np.sort(np.abs(errors));cdf=np.arange(1,len(sorted_e)+1)/len(sorted_e)
ax.plot(sorted_e,cdf,color=PALETTE["a2"],linewidth=2)
ax.axvline(sorted_e[int(0.9*len(sorted_e))],color=PALETTE["a3"],linestyle="--",linewidth=1.2,label="90th pct")
setup_ax(ax,"CDF of Absolute Errors","|Residual|","Cumulative Proportion")
ax.legend(facecolor=PALETTE["card"],labelcolor=PALETTE["text"],fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"phase4_error_distribution.png"),dpi=150,bbox_inches="tight",facecolor=PALETTE["bg"])
plt.close()
print("Plot: error distribution saved")

# Plot 5: CV comparison
x=np.arange(len(names));w=0.35
fig,ax=plt.subplots(figsize=(12,5))
fig.patch.set_facecolor(PALETTE["bg"])
ax.set_facecolor(PALETTE["card"])
b1=ax.bar(x-w/2,r2s,w,label="Test R2",color=PALETTE["a1"],edgecolor=PALETTE["border"])
b2=ax.bar(x+w/2,cv_r2s,w,label="CV R2 (5-fold)",color=PALETTE["a2"],edgecolor=PALETTE["border"])
ax.bar_label(b1,fmt="%.3f",color=PALETTE["text"],fontsize=8,padding=2)
ax.bar_label(b2,fmt="%.3f",color=PALETTE["text"],fontsize=8,padding=2)
ax.set_xticks(x);ax.set_xticklabels(names,rotation=20,ha="right",fontsize=9)
ax.set_ylim(max(0,min(r2s+cv_r2s)-0.05),1.05)
ax.legend(facecolor=PALETTE["card"],labelcolor=PALETTE["text"])
setup_ax(ax,"Test R2 vs 5-Fold CV R2 per Model","Model","R2")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR,"phase4_cv_comparison.png"),dpi=150,bbox_inches="tight",facecolor=PALETTE["bg"])
plt.close()
print("Plot: CV comparison saved")

print("\nPhase 4 COMPLETE.")
