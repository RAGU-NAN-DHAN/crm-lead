"""
PAA-NSGA-II — Phase 3: SLA Breach Prediction on BPI Challenge 2014
===================================================================
Dataset: Rabobank Nederland Group ICT — real incident management data
Target:  SLA Breach (binary) — did the case exceed its expected handle time?

Plan:
  1. Load & merge the 3 BPI 2014 files
  2. Engineer features from real operational data (with written justifications)
  3. Train XGBoost (primary), LightGBM, CatBoost, Random Forest
  4. Feature combination experiment: 5 → 8 → 12 → all features
  5. Full validation: 5-fold stratified CV, precision, recall, F1, confusion matrix, ROC-AUC
  6. SHAP analysis on best model
  7. Output results tables + plots

Outputs go to crm-project/results_bpi/
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, classification_report,
    roc_curve, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# ── paths ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "bpi2014"
RESULTS_DIR = BASE_DIR / "results_bpi"
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)

# =====================================================================
# 1.  LOAD & MERGE DATA
# =====================================================================
print("Loading BPI Challenge 2014 data...")

inc = pd.read_csv(DATA_DIR / "Detail_Incident.csv", sep=";", encoding="latin1")
act = pd.read_csv(DATA_DIR / "Detail_Incident_Activity.csv", sep=";", encoding="latin1")
intr = pd.read_csv(DATA_DIR / "Detail_Interaction.csv", sep=";", encoding="latin1")

print(f"  Incidents:    {len(inc):,} rows")
print(f"  Activities:   {len(act):,} rows")
print(f"  Interactions: {len(intr):,} rows")

# Fix Handle Time (uses comma as decimal separator)
inc["Handle_Time_Hours"] = pd.to_numeric(
    inc["Handle Time (Hours)"].astype(str).str.replace(",", "."),
    errors="coerce"
)

# Parse timestamps
for col in ["Open Time", "Resolved Time", "Close Time"]:
    inc[col] = pd.to_datetime(inc[col], format="mixed", dayfirst=False, errors="coerce")

act["DateStamp"] = pd.to_datetime(act["DateStamp"], format="mixed", dayfirst=True, errors="coerce")

# =====================================================================
# 2.  FEATURE ENGINEERING (with justifications)
# =====================================================================
print("\nEngineering features...")

# ── 2.1  Target: SLA Breach ──────────────────────────────────────────
# JUSTIFICATION: The system records Handle Time measured on HP Service Manager's
# own SLA clock. We define breach as exceeding the priority-specific median
# handle time by a factor, which simulates a realistic SLA threshold per
# priority band. This avoids inventing an arbitrary SLA and uses the data's
# own operational norms.
priority_medians = inc.groupby("Priority")["Handle_Time_Hours"].median()
print(f"\nMedian handle time by priority:")
for p, m in priority_medians.items():
    print(f"  Priority {p}: {m:.2f} hours")

# SLA threshold: 2x the median for that priority level (captures the slow tail)
sla_thresholds = {p: m * 2.0 for p, m in priority_medians.items()}
inc["SLA_Threshold_Hours"] = inc["Priority"].map(sla_thresholds)
inc["SLA_Breached"] = (inc["Handle_Time_Hours"] > inc["SLA_Threshold_Hours"]).astype(int)

# Drop rows with missing handle time or priority
df = inc[inc["Handle_Time_Hours"].notna() & inc["Priority"].notna()].copy()
print(f"\nAfter cleaning: {len(df):,} incidents")
print(f"SLA breach rate: {df['SLA_Breached'].mean():.1%}")

# ── 2.2  Features from Activity Log ──────────────────────────────────

# Assignment events per incident
assign_events = act[act["IncidentActivity_Type"].isin(["Assignment", "Reassignment"])]

# First assignment group per incident
# JUSTIFICATION: The initial assignment group is the operational decision point.
# Who the case is first routed to affects everything downstream.
first_assign = (assign_events.sort_values("DateStamp")
                .groupby("Incident ID")["Assignment Group"]
                .first()
                .reset_index()
                .rename(columns={"Assignment Group": "First_Assignment_Group"}))

df = df.merge(first_assign, left_on="Incident ID", right_on="Incident ID", how="left")

# Number of distinct assignment groups touched
# JUSTIFICATION: More groups = more handoffs = complexity indicator
groups_touched = (assign_events.groupby("Incident ID")["Assignment Group"]
                 .nunique()
                 .reset_index()
                 .rename(columns={"Assignment Group": "Num_Groups_Touched"}))
df = df.merge(groups_touched, on="Incident ID", how="left")
df["Num_Groups_Touched"] = df["Num_Groups_Touched"].fillna(0).astype(int)

# Total activity events per incident
# JUSTIFICATION: More events = higher complexity, more touches needed
event_counts = (act.groupby("Incident ID")
                .size()
                .reset_index(name="Total_Activity_Events"))
df = df.merge(event_counts, on="Incident ID", how="left")
df["Total_Activity_Events"] = df["Total_Activity_Events"].fillna(0).astype(int)

# Time from open to first assignment (assignment delay)
# JUSTIFICATION: How long a case sits unassigned is a key predictor of breach
first_assign_time = (assign_events.sort_values("DateStamp")
                     .groupby("Incident ID")["DateStamp"]
                     .first()
                     .reset_index()
                     .rename(columns={"DateStamp": "First_Assignment_Time"}))
df = df.merge(first_assign_time, on="Incident ID", how="left")
df["Assignment_Delay_Hours"] = (
    (df["First_Assignment_Time"] - df["Open Time"])
    .dt.total_seconds() / 3600
).clip(lower=0)
df["Assignment_Delay_Hours"] = df["Assignment_Delay_Hours"].fillna(0)

# ── 2.3  Assignment Group Performance Features ───────────────────────
# JUSTIFICATION: Historical group performance (resolution rate, speed, workload)
# is exactly what a decision engine should consider — Section 4.3 Group B.

# Group historical stats (computed over entire dataset as training signal)
group_stats = df.groupby("First_Assignment_Group").agg(
    Group_Case_Count=("Incident ID", "count"),
    Group_Avg_Handle_Time=("Handle_Time_Hours", "mean"),
    Group_Median_Handle_Time=("Handle_Time_Hours", "median"),
    Group_Breach_Rate=("SLA_Breached", "mean"),
    Group_Reassignment_Rate=("# Reassignments", lambda x: (x > 0).mean()),
    Group_Avg_Reassignments=("# Reassignments", "mean"),
).reset_index()

df = df.merge(group_stats, on="First_Assignment_Group", how="left")

# ── 2.4  Interaction features ────────────────────────────────────────
# JUSTIFICATION: First Call Resolution and interaction count are real success
# measures from the interaction log

# Aggregate interaction info per related incident
intr_agg = (intr.groupby("Related Incident")
            .agg(
                Interaction_Count=("Interaction ID", "count"),
                FCR_Rate=("First Call Resolution", lambda x: (x == "Y").mean()),
                Avg_Interaction_Handle_Secs=("Handle Time (secs)", "mean"),
            )
            .reset_index())

df = df.merge(intr_agg, left_on="Incident ID", right_on="Related Incident", how="left")
df["Interaction_Count"] = df["Interaction_Count"].fillna(0).astype(int)
df["FCR_Rate"] = df["FCR_Rate"].fillna(0)
df["Avg_Interaction_Handle_Secs"] = df["Avg_Interaction_Handle_Secs"].fillna(0)

# ── 2.5  Temporal features ───────────────────────────────────────────
# JUSTIFICATION: Time-of-day and day-of-week capture shift patterns,
# staffing levels, and workload cycles — standard in operational ML

df["Open_Hour"] = df["Open Time"].dt.hour
df["Open_DayOfWeek"] = df["Open Time"].dt.dayofweek  # 0=Monday
df["Is_Weekend"] = (df["Open_DayOfWeek"] >= 5).astype(int)
df["Is_Business_Hours"] = ((df["Open_Hour"] >= 8) & (df["Open_Hour"] <= 18)).astype(int)

# Month (seasonal load)
df["Open_Month"] = df["Open Time"].dt.month

# ── 2.6  Case complexity features ────────────────────────────────────
# JUSTIFICATION: Related incidents/changes indicate systemic issues;
# a case tied to a bigger problem takes longer

df["Num_Related_Incidents"] = pd.to_numeric(df["# Related Incidents"], errors="coerce").fillna(0)
df["Num_Related_Changes"] = pd.to_numeric(df["# Related Changes"], errors="coerce").fillna(0)
df["Num_Related_Interactions"] = pd.to_numeric(df["# Related Interactions"], errors="coerce").fillna(0)
df["Has_Related_Change"] = (df["Num_Related_Changes"] > 0).astype(int)
df["Has_Reopen"] = df["Reopen Time"].notna().astype(int)

# ── 2.7  Queue state at open time (simulated from actual timestamps) ─
# JUSTIFICATION: How many cases are already open when this one arrives
# directly measures system load — a key SLA risk factor

df_sorted = df.sort_values("Open Time")
open_times = df_sorted["Open Time"].values
resolved_times = df_sorted["Resolved Time"].values

# For each case, count how many cases are currently open (opened before, not yet resolved)
# This is O(n) with a sweep
queue_lengths = []
open_set = []
resolve_idx = 0

for i, (ot, rt) in enumerate(zip(open_times, resolved_times)):
    # Remove resolved cases from open_set
    open_set = [r for r in open_set if r > ot or pd.isna(r)]
    queue_lengths.append(len(open_set))
    # Add this case's resolve time
    if pd.notna(rt):
        open_set.append(rt)

df_sorted["Queue_Length_At_Open"] = queue_lengths
df = df.merge(
    df_sorted[["Incident ID", "Queue_Length_At_Open"]],
    on="Incident ID", how="left"
)

# =====================================================================
# 3.  FEATURE SETS (for the combination experiment)
# =====================================================================

# All engineered features (what the model sees)
ALL_FEATURES = [
    # Case attributes (from incident record)
    "Priority",                    # 1.  Real: impact × urgency
    "Impact",                      # 2.  Real: business impact level
    "Urgency",                     # 3.  Real: how urgent
    "# Reassignments",             # 4.  Real: number of reassignments
    "Num_Related_Incidents",       # 5.  Real: linked incidents

    # Temporal
    "Open_Hour",                   # 6.  Derived: hour of day
    "Open_DayOfWeek",              # 7.  Derived: day of week
    "Is_Weekend",                  # 8.  Derived: weekend flag

    # Operational (from activity log)
    "Assignment_Delay_Hours",      # 9.  Derived: time to first assignment
    "Num_Groups_Touched",          # 10. Derived: handoff count
    "Total_Activity_Events",       # 11. Derived: event complexity
    "Queue_Length_At_Open",        # 12. Derived: system load at intake

    # Group performance
    "Group_Avg_Handle_Time",       # 13. Derived: group's historical speed
    "Group_Breach_Rate",           # 14. Derived: group's historical breach %
    "Group_Reassignment_Rate",     # 15. Derived: group's reassignment tendency
    "Group_Case_Count",            # 16. Derived: group size/experience

    # Interaction features
    "Interaction_Count",           # 17. Real: number of interactions
    "FCR_Rate",                    # 18. Real: first call resolution rate

    # Case complexity
    "Num_Related_Changes",         # 19. Real: linked changes
    "Has_Reopen",                  # 20. Real: was case reopened
    "Is_Business_Hours",           # 21. Derived: opened during business hours
    "Open_Month",                  # 22. Derived: seasonal
    "Num_Related_Interactions",    # 23. Real: interaction count from incident
    "Has_Related_Change",          # 24. Derived: linked to a change request

    # CI attributes
    "CI_Type_Encoded",             # 25. Real: configuration item type
    "Category_Encoded",            # 26. Real: incident category
]

# Encode categoricals
le_ci = LabelEncoder()
df["CI_Type_Encoded"] = le_ci.fit_transform(df["CI Type (aff)"].fillna("unknown").astype(str))

le_cat = LabelEncoder()
df["Category_Encoded"] = le_cat.fit_transform(df["Category"].fillna("unknown").astype(str))

le_group = LabelEncoder()
df["First_Group_Encoded"] = le_group.fit_transform(
    df["First_Assignment_Group"].fillna("unknown").astype(str)
)

# Feature subsets for the combination experiment
# Matching prior work's ~5 parameters
FEATURES_5 = [
    "Priority", "Impact", "Urgency", "# Reassignments", "Num_Related_Incidents"
]

FEATURES_8 = FEATURES_5 + [
    "Open_Hour", "Open_DayOfWeek", "Is_Weekend"
]

FEATURES_12 = FEATURES_8 + [
    "Assignment_Delay_Hours", "Num_Groups_Touched",
    "Total_Activity_Events", "Queue_Length_At_Open"
]

FEATURES_ALL = ALL_FEATURES

# =====================================================================
# 4.  PREPARE FINAL DATASET
# =====================================================================

# Drop rows with NaN in any feature
feature_cols = FEATURES_ALL
target_col = "SLA_Breached"

df_model = df[feature_cols + [target_col, "Incident ID"]].dropna().copy()
print(f"\nFinal modelling dataset: {len(df_model):,} rows x {len(feature_cols)} features")
print(f"SLA breach rate: {df_model[target_col].mean():.1%}")
print(f"Class distribution: {df_model[target_col].value_counts().to_dict()}")

X = df_model[feature_cols]
y = df_model[target_col]

# Hold out 20% as final test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# =====================================================================
# 5.  MODEL COMPARISON (all 4 algorithms, full features)
# =====================================================================

def get_models():
    return {
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            eval_metric="logloss", random_state=SEED, verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            is_unbalance=True,
            random_state=SEED, verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            auto_class_weights="Balanced",
            random_seed=SEED, verbose=0,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=12,
            class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        ),
    }


def run_model_comparison(X_tr, X_te, y_tr, y_te, feature_label):
    """Train all 4 models, return results + best model."""
    print(f"\n{'='*60}")
    print(f"  Model Comparison — {feature_label} ({X_tr.shape[1]} features)")
    print(f"  Train: {len(X_tr):,}  |  Test: {len(X_te):,}")
    print(f"{'='*60}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []

    for name, model in get_models().items():
        # 5-fold CV
        fold_aucs = []
        for tr_idx, va_idx in skf.split(X_tr, y_tr):
            Xf_tr, Xf_va = X_tr.iloc[tr_idx], X_tr.iloc[va_idx]
            yf_tr, yf_va = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]
            model.fit(Xf_tr, yf_tr)
            proba = model.predict_proba(Xf_va)[:, 1]
            fold_aucs.append(roc_auc_score(yf_va, proba))

        cv_auc = np.mean(fold_aucs)
        cv_std = np.std(fold_aucs)

        # Refit on full train, evaluate on held-out test
        model.fit(X_tr, y_tr)
        test_proba = model.predict_proba(X_te)[:, 1]
        test_pred = model.predict(X_te)

        acc  = accuracy_score(y_te, test_pred)
        prec = precision_score(y_te, test_pred, zero_division=0)
        rec  = recall_score(y_te, test_pred, zero_division=0)
        f1   = f1_score(y_te, test_pred, zero_division=0)
        auc  = roc_auc_score(y_te, test_proba)
        brier = brier_score_loss(y_te, test_proba)

        results.append({
            "Model": name,
            "Features": feature_label,
            "Num_Features": X_tr.shape[1],
            "CV_AUC_mean": round(cv_auc, 4),
            "CV_AUC_std": round(cv_std, 4),
            "Test_Accuracy": round(acc, 4),
            "Test_Precision": round(prec, 4),
            "Test_Recall": round(rec, 4),
            "Test_F1": round(f1, 4),
            "Test_AUC": round(auc, 4),
            "Test_Brier": round(brier, 4),
            "_model": model,
            "_test_proba": test_proba,
        })
        print(f"  {name:15s}  CV-AUC={cv_auc:.4f}+/-{cv_std:.4f}  "
              f"Test-AUC={auc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")

    return results


# Run comparison with ALL features
all_results = run_model_comparison(X_train, X_test, y_train, y_test, "All_26_features")

# Pick best model
best = max(all_results, key=lambda r: r["Test_AUC"])
print(f"\n  ** Best: {best['Model']} (AUC={best['Test_AUC']:.4f}) **")

# =====================================================================
# 6.  FEATURE COMBINATION EXPERIMENT (XGBoost only)
# =====================================================================
print("\n" + "="*60)
print("  FEATURE COMBINATION EXPERIMENT (XGBoost)")
print("="*60)

combo_results = []
for label, feat_list in [
    ("5_basic", FEATURES_5),
    ("8_temporal", FEATURES_8),
    ("12_operational", FEATURES_12),
    ("26_all", FEATURES_ALL),
]:
    Xtr_sub = X_train[feat_list]
    Xte_sub = X_test[feat_list]

    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        eval_metric="logloss", random_state=SEED, verbosity=0,
    )

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_aucs = []
    for tr_idx, va_idx in skf.split(Xtr_sub, y_train):
        xgb.fit(Xtr_sub.iloc[tr_idx], y_train.iloc[tr_idx])
        proba = xgb.predict_proba(Xtr_sub.iloc[va_idx])[:, 1]
        fold_aucs.append(roc_auc_score(y_train.iloc[va_idx], proba))

    cv_auc = np.mean(fold_aucs)

    # Refit on full train
    xgb.fit(Xtr_sub, y_train)
    test_proba = xgb.predict_proba(Xte_sub)[:, 1]
    test_pred = xgb.predict(Xte_sub)

    acc  = accuracy_score(y_test, test_pred)
    prec = precision_score(y_test, test_pred, zero_division=0)
    rec  = recall_score(y_test, test_pred, zero_division=0)
    f1   = f1_score(y_test, test_pred, zero_division=0)
    auc  = roc_auc_score(y_test, test_proba)

    combo_results.append({
        "Feature_Set": label,
        "Num_Features": len(feat_list),
        "Features": ", ".join(feat_list),
        "CV_AUC": round(cv_auc, 4),
        "Test_AUC": round(auc, 4),
        "Test_Accuracy": round(acc, 4),
        "Test_Precision": round(prec, 4),
        "Test_Recall": round(rec, 4),
        "Test_F1": round(f1, 4),
        "_model": xgb,
        "_test_proba": test_proba,
    })
    print(f"  {label:20s}  ({len(feat_list):2d} feats)  CV-AUC={cv_auc:.4f}  "
          f"Test-AUC={auc:.4f}  F1={f1:.4f}")

# =====================================================================
# 7.  FULL VALIDATION — BEST MODEL (XGBoost, all features)
# =====================================================================
print("\n" + "="*60)
print("  FULL VALIDATION — XGBoost (all features)")
print("="*60)

best_xgb = combo_results[-1]["_model"]  # all features
best_proba = combo_results[-1]["_test_proba"]
best_pred = (best_proba >= 0.5).astype(int)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, best_pred, target_names=["No Breach", "SLA Breach"]))

# Confusion matrix
cm = confusion_matrix(y_test, best_pred)
print(f"Confusion Matrix:")
print(f"  TN={cm[0,0]:5d}   FP={cm[0,1]:5d}")
print(f"  FN={cm[1,0]:5d}   TP={cm[1,1]:5d}")

# =====================================================================
# 8.  CALIBRATION
# =====================================================================
print("\nCalibrating model (isotonic)...")
cal_xgb = CalibratedClassifierCV(best_xgb, method="isotonic", cv=3)
cal_xgb.fit(X_train, y_train)
cal_proba = cal_xgb.predict_proba(X_test)[:, 1]

raw_brier = brier_score_loss(y_test, best_proba)
cal_brier = brier_score_loss(y_test, cal_proba)
print(f"  Raw Brier:        {raw_brier:.4f}")
print(f"  Calibrated Brier: {cal_brier:.4f}")

# =====================================================================
# 9.  PLOTS
# =====================================================================

# --- 9.1 ROC curves for all 4 models ---
fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
for r in all_results:
    fpr, tpr, _ = roc_curve(y_test, r["_test_proba"])
    ax_roc.plot(fpr, tpr, label=f"{r['Model']} (AUC={r['Test_AUC']:.3f})")
ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("SLA Breach Prediction - ROC Curves (All Models)")
ax_roc.legend()
fig_roc.tight_layout()
fig_roc.savefig(RESULTS_DIR / "roc_all_models.png", dpi=150)
plt.close(fig_roc)
print("\nSaved roc_all_models.png")

# --- 9.2 Feature combination AUC comparison ---
fig_combo, ax_combo = plt.subplots(figsize=(8, 5))
labels = [r["Feature_Set"] for r in combo_results]
aucs = [r["Test_AUC"] for r in combo_results]
bars = ax_combo.bar(labels, aucs, color=["#2196F3", "#4CAF50", "#FF9800", "#F44336"])
for bar, auc in zip(bars, aucs):
    ax_combo.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                  f"{auc:.4f}", ha="center", va="bottom", fontweight="bold")
ax_combo.set_ylabel("Test AUC")
ax_combo.set_title("XGBoost Performance by Feature Set Size")
ax_combo.set_ylim(min(aucs) - 0.05, max(aucs) + 0.03)
fig_combo.tight_layout()
fig_combo.savefig(RESULTS_DIR / "feature_combo_comparison.png", dpi=150)
plt.close(fig_combo)
print("Saved feature_combo_comparison.png")

# --- 9.3 Confusion matrix heatmap ---
fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
im = ax_cm.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax_cm.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                   fontsize=16, color="white" if cm[i,j] > cm.max()/2 else "black")
ax_cm.set_xticks([0, 1])
ax_cm.set_yticks([0, 1])
ax_cm.set_xticklabels(["No Breach", "SLA Breach"])
ax_cm.set_yticklabels(["No Breach", "SLA Breach"])
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix - XGBoost (All Features)")
fig_cm.colorbar(im)
fig_cm.tight_layout()
fig_cm.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
plt.close(fig_cm)
print("Saved confusion_matrix.png")

# --- 9.4 Calibration plot ---
fig_cal, axes_cal = plt.subplots(1, 2, figsize=(14, 5))
for ax, proba, label in [
    (axes_cal[0], best_proba, "Raw XGBoost"),
    (axes_cal[1], cal_proba, "Calibrated (Isotonic)"),
]:
    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)
    ax.plot(prob_pred, prob_true, "o-", label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"SLA Breach - {label}")
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
fig_cal.tight_layout()
fig_cal.savefig(RESULTS_DIR / "calibration_plot.png", dpi=150)
plt.close(fig_cal)
print("Saved calibration_plot.png")

# --- 9.5 SHAP analysis ---
print("\nRunning SHAP analysis...")
try:
    explainer = shap.TreeExplainer(best_xgb)
    shap_values = explainer.shap_values(X_test.iloc[:500])

    # Global importance bar
    fig_shap, ax_shap = plt.subplots(figsize=(10, 8))
    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    feat_names = [FEATURES_ALL[i] for i in sorted_idx]
    ax_shap.barh(range(len(sorted_idx)), mean_abs[sorted_idx][::-1])
    ax_shap.set_yticks(range(len(sorted_idx)))
    ax_shap.set_yticklabels(feat_names[::-1])
    ax_shap.set_xlabel("Mean |SHAP value|")
    ax_shap.set_title("SLA Breach - Feature Importance (SHAP)")
    fig_shap.tight_layout()
    fig_shap.savefig(RESULTS_DIR / "shap_importance.png", dpi=150)
    plt.close(fig_shap)
    print("Saved shap_importance.png")

    # Beeswarm
    fig_bee = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test.iloc[:500],
                      feature_names=FEATURES_ALL, show=False, max_display=20)
    plt.title("SLA Breach - SHAP Beeswarm")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_beeswarm.png", dpi=150)
    plt.close()
    print("Saved shap_beeswarm.png")
except Exception as e:
    print(f"SHAP failed: {e}")

# =====================================================================
# 10.  SAVE RESULTS TABLES
# =====================================================================

# Model comparison table
comp_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                         for r in all_results])
comp_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
print("\nSaved model_comparison.csv")

# Feature combo table
combo_df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                          for r in combo_results])
combo_df.to_csv(RESULTS_DIR / "feature_combination_results.csv", index=False)
print("Saved feature_combination_results.csv")

# Feature justification table
justifications = pd.DataFrame({
    "Feature": FEATURES_ALL,
    "Source": [
        "Real", "Real", "Real", "Real", "Real",           # 1-5 case
        "Derived", "Derived", "Derived",                    # 6-8 temporal
        "Derived", "Derived", "Derived", "Derived",         # 9-12 operational
        "Derived", "Derived", "Derived", "Derived",         # 13-16 group
        "Real", "Real",                                     # 17-18 interaction
        "Real", "Real", "Derived", "Derived", "Real", "Derived",  # 19-24 complexity
        "Real", "Real",                                     # 25-26 CI/category
    ],
    "Justification": [
        "Impact x Urgency priority from HP SM SLA clock",
        "Business impact level — drives SLA stringency",
        "How urgent the caller reports the issue",
        "Number of times case was rerouted — operational cost signal",
        "Linked incidents — indicates systemic issue",
        "Hour of day — captures shift/staffing patterns",
        "Day of week — weekend vs weekday staffing",
        "Weekend flag — reduced staffing, longer queues",
        "Hours from case open to first assignment — unassigned wait",
        "Distinct groups the case passed through — handoff complexity",
        "Total activity events — case complexity indicator",
        "Cases currently open when this one arrives — system load",
        "Assignment group's historical avg handle time",
        "Assignment group's historical SLA breach rate",
        "Assignment group's tendency to reassign cases",
        "How many cases the group has handled — experience proxy",
        "Number of customer interactions on the case",
        "First Call Resolution rate for this case's interactions",
        "Linked change requests — indicates scope",
        "Whether case was reopened — resolution quality signal",
        "Opened during 8am-6pm — staffing availability",
        "Month — seasonal workload variation",
        "Number of related interactions from incident record",
        "Whether a change request is linked",
        "Configuration item type — hardware, software, app, etc.",
        "Incident category — incident, request, complaint",
    ]
})
justifications.to_csv(RESULTS_DIR / "feature_justifications.csv", index=False)
print("Saved feature_justifications.csv")

# =====================================================================
# 11.  SUMMARY
# =====================================================================
print("\n" + "="*60)
print("  FINAL SUMMARY")
print("="*60)
print(f"  Dataset:    BPI Challenge 2014 (Rabobank)")
print(f"  Rows:       {len(df_model):,}")
print(f"  Target:     SLA Breach ({df_model[target_col].mean():.1%} positive)")
print(f"  Best model: {best['Model']} (AUC = {best['Test_AUC']:.4f})")
print(f"\n  Feature Combination Results (XGBoost):")
for r in combo_results:
    print(f"    {r['Feature_Set']:20s}  {r['Num_Features']:2d} feats  "
          f"AUC={r['Test_AUC']:.4f}  F1={r['Test_F1']:.4f}")
print(f"\n  All outputs in: {RESULTS_DIR}")
print("="*60)
