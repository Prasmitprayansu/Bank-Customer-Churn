# =============================================================================
# BANK CUSTOMER CHURN PREDICTION
# Stacking Ensemble: XGBoost + LightGBM + CatBoost
# Improvements: SMOTE, Feature Engineering, Deep EDA, Threshold Tuning, SHAP
# =============================================================================
# SETUP — run this once in your terminal before running the script:
#   pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
#           catboost category_encoders imbalanced-learn shap
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    precision_recall_curve, roc_curve
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import shap


# =============================================================================
# 1. LOAD DATA
# =============================================================================
# Download the dataset from:
# https://www.kaggle.com/datasets/abbas829/bank-customer-churn
# Place Bank_Churn.csv in the same folder as this script.

df = pd.read_csv("Bank_Churn.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicates: {df.duplicated().sum()}")
print(f"\nClass distribution:\n{df['Exited'].value_counts()}")
print(f"Churn rate: {df['Exited'].mean():.2%}")


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\n" + "=" * 60)
print("RUNNING EDA — check the plots folder")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Bank Customer Churn — EDA", fontsize=16, fontweight="bold")

# 2a. Overall churn distribution
churn_counts = df["Exited"].value_counts()
axes[0, 0].bar(["Not Churned", "Churned"], churn_counts.values,
               color=["#378ADD", "#D85A30"], edgecolor="white", linewidth=1.5)
axes[0, 0].set_title("Churn Distribution")
axes[0, 0].set_ylabel("Count")
for i, v in enumerate(churn_counts.values):
    axes[0, 0].text(i, v + 50, f"{v}\n({v/len(df):.1%})",
                    ha="center", fontsize=10)

# 2b. Churn rate by Geography
geo_churn = df.groupby("Geography")["Exited"].mean().sort_values(ascending=False)
bars = axes[0, 1].bar(geo_churn.index, geo_churn.values * 100,
                       color=["#D85A30", "#F0997B", "#378ADD"], edgecolor="white")
axes[0, 1].set_title("Churn Rate by Geography")
axes[0, 1].set_ylabel("Churn Rate (%)")
for bar, val in zip(bars, geo_churn.values):
    axes[0, 1].text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3, f"{val:.1%}", ha="center", fontsize=10)

# 2c. Churn rate by Number of Products
prod_churn = df.groupby("NumOfProducts")["Exited"].mean()
axes[0, 2].bar(prod_churn.index.astype(str), prod_churn.values * 100,
               color="#5DCAA5", edgecolor="white")
axes[0, 2].set_title("Churn Rate by Number of Products")
axes[0, 2].set_xlabel("Number of Products")
axes[0, 2].set_ylabel("Churn Rate (%)")
for i, (idx, val) in enumerate(prod_churn.items()):
    axes[0, 2].text(i, val * 100 + 0.5, f"{val:.1%}", ha="center", fontsize=10)

# 2d. Age distribution — churned vs not churned
df[df["Exited"] == 0]["Age"].plot.hist(
    ax=axes[1, 0], bins=30, alpha=0.6, color="#378ADD", label="Not Churned")
df[df["Exited"] == 1]["Age"].plot.hist(
    ax=axes[1, 0], bins=30, alpha=0.6, color="#D85A30", label="Churned")
axes[1, 0].set_title("Age Distribution: Churned vs Not Churned")
axes[1, 0].set_xlabel("Age")
axes[1, 0].set_ylabel("Count")
axes[1, 0].legend()

# 2e. Balance distribution — churned vs not churned
df[df["Exited"] == 0]["Balance"].plot.hist(
    ax=axes[1, 1], bins=30, alpha=0.6, color="#378ADD", label="Not Churned")
df[df["Exited"] == 1]["Balance"].plot.hist(
    ax=axes[1, 1], bins=30, alpha=0.6, color="#D85A30", label="Churned")
axes[1, 1].set_title("Balance Distribution: Churned vs Not Churned")
axes[1, 1].set_xlabel("Balance")
axes[1, 1].set_ylabel("Count")
axes[1, 1].legend()

# 2f. Churn rate by IsActiveMember
active_churn = df.groupby("IsActiveMember")["Exited"].mean()
axes[1, 2].bar(["Inactive", "Active"], active_churn.values * 100,
               color=["#D85A30", "#378ADD"], edgecolor="white")
axes[1, 2].set_title("Churn Rate: Active vs Inactive Members")
axes[1, 2].set_ylabel("Churn Rate (%)")
for i, val in enumerate(active_churn.values):
    axes[1, 2].text(i, val * 100 + 0.3, f"{val:.1%}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("EDA plot saved as eda_plots.png")

# Correlation heatmap
plt.figure(figsize=(12, 9))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm",
            fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.show()
print("Correlation matrix saved as correlation_matrix.png")


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

data = df.copy()

# New feature: balance-to-salary ratio (high ratio = more reliant on this bank)
data["Balance_Salary_Ratio"] = data["Balance"] / (data["EstimatedSalary"] + 1)

# New feature: is the customer's balance zero?
data["Zero_Balance"] = (data["Balance"] == 0).astype(int)

# New feature: age group bucket
data["Age_Group"] = pd.cut(
    data["Age"],
    bins=[0, 30, 40, 50, 60, 100],
    labels=["<30", "30-40", "40-50", "50-60", "60+"]
)

# New feature: products per year of tenure
data["Products_Per_Tenure"] = data["NumOfProducts"] / (data["Tenure"] + 1)

# New feature: credit score bucket
data["CreditScore_Bucket"] = pd.cut(
    data["CreditScore"],
    bins=[0, 579, 669, 739, 799, 850],
    labels=["Poor", "Fair", "Good", "Very Good", "Exceptional"]
)

print("New features added:")
print("  - Balance_Salary_Ratio")
print("  - Zero_Balance")
print("  - Age_Group")
print("  - Products_Per_Tenure")
print("  - CreditScore_Bucket")
print(f"\nDataset shape after feature engineering: {data.shape}")


# =============================================================================
# 4. PREPROCESSING
# =============================================================================
print("\n" + "=" * 60)
print("PREPROCESSING")
print("=" * 60)

# Drop ID columns and target
X = data.drop(["CustomerId", "Surname", "Exited"], axis=1)
y = data["Exited"]

# Train/test split — stratified to preserve class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
print(f"\nClass distribution in y_train:\n{y_train.value_counts()}")

# Target encoding for categorical columns
# Note: Gender is binary — label-encode it; Geography has 3 values — target encode
categorical_cols = ["Geography", "Gender", "Age_Group", "CreditScore_Bucket"]
encoder = TargetEncoder(cols=categorical_cols)
X_train = encoder.fit_transform(X_train, y_train)  # fit ONLY on train
X_test = encoder.transform(X_test)                 # transform test separately


# =============================================================================
# 5. SMOTE — FIX CLASS IMBALANCE
# =============================================================================
print("\n" + "=" * 60)
print("SMOTE — FIXING CLASS IMBALANCE")
print("=" * 60)

from collections import Counter
print(f"Before SMOTE: {Counter(y_train)}")

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"After SMOTE:  {Counter(y_train_sm)}")
print("NOTE: SMOTE is applied ONLY on training data. Test set is untouched.")


# =============================================================================
# 6. STACKING ENSEMBLE MODEL
# =============================================================================
print("\n" + "=" * 60)
print("TRAINING STACKING ENSEMBLE")
print("=" * 60)

estimators = [
    ("xgb", xgb.XGBClassifier(
        n_estimators=721,
        max_depth=4,
        learning_rate=0.04493087550798207,
        subsample=0.9333887317969405,
        colsample_bytree=0.7100069311646524,
        gamma=0.28125146754150937,
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )),
    ("lgbm", lgb.LGBMClassifier(
        n_estimators=998,
        max_depth=7,
        learning_rate=0.05841823930928472,
        subsample=0.7997178697624011,
        colsample_bytree=0.7489077535165226,
        reg_alpha=0.40028814506270205,
        reg_lambda=0.2782201112963498,
        verbose=-1,
        random_state=42
    )),
    ("catboost", CatBoostClassifier(
        iterations=316,
        depth=8,
        learning_rate=0.18288634764672076,
        l2_leaf_reg=6.759898410871825,
        random_state=42,
        verbose=0
    ))
]

final_estimator = LogisticRegression(
    C=0.9417973749284854,
    random_state=42,
    solver="liblinear"
)

stacked_model = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=5,
    n_jobs=-1
)

print("Training... (this takes 2-4 minutes)")
stacked_model.fit(X_train_sm, y_train_sm)
print("Training complete!")

y_proba = stacked_model.predict_proba(X_test)[:, 1]
y_pred_default = stacked_model.predict(X_test)


# =============================================================================
# 7. THRESHOLD TUNING
# =============================================================================
print("\n" + "=" * 60)
print("THRESHOLD TUNING")
print("=" * 60)

precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Default threshold (0.50) F1: {f1_score(y_test, y_pred_default):.4f}")
print(f"Optimal threshold found:     {best_threshold:.4f}")

y_pred_tuned = (y_proba >= best_threshold).astype(int)
print(f"Tuned threshold F1:          {f1_score(y_test, y_pred_tuned):.4f}")

# Precision-recall curve plot
plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions[:-1], label="Precision", color="#378ADD")
plt.plot(thresholds, recalls[:-1], label="Recall", color="#D85A30")
plt.plot(thresholds, f1_scores[:-1], label="F1 Score", color="#1D9E75", linestyle="--")
plt.axvline(best_threshold, color="gray", linestyle=":", label=f"Best threshold ({best_threshold:.2f})")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision, Recall, F1 vs Threshold")
plt.legend()
plt.tight_layout()
plt.savefig("threshold_tuning.png", dpi=150, bbox_inches="tight")
plt.show()
print("Threshold plot saved as threshold_tuning.png")


# =============================================================================
# 8. EVALUATION
# =============================================================================
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print("\n--- Default Threshold (0.50) ---")
print(classification_report(y_test, y_pred_default,
                             target_names=["Not Churned", "Churned"]))

print(f"\n--- Tuned Threshold ({best_threshold:.2f}) ---")
print(classification_report(y_test, y_pred_tuned,
                             target_names=["Not Churned", "Churned"]))

print("\nSummary:")
print(f"{'Metric':<20} {'Default (0.50)':>16} {'Tuned ({:.2f})'.format(best_threshold):>16}")
print("-" * 54)
metrics = {
    "Accuracy":  (accuracy_score(y_test, y_pred_default),  accuracy_score(y_test, y_pred_tuned)),
    "Precision": (precision_score(y_test, y_pred_default), precision_score(y_test, y_pred_tuned)),
    "Recall":    (recall_score(y_test, y_pred_default),    recall_score(y_test, y_pred_tuned)),
    "F1 Score":  (f1_score(y_test, y_pred_default),        f1_score(y_test, y_pred_tuned)),
    "ROC AUC":   (roc_auc_score(y_test, y_proba),          roc_auc_score(y_test, y_proba)),
}
for name, (d, t) in metrics.items():
    print(f"{name:<20} {d:>16.4f} {t:>16.4f}")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="#D85A30", lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight")
plt.show()
print("ROC curve saved as roc_curve.png")


# =============================================================================
# 9. SHAP EXPLAINABILITY
# =============================================================================
print("\n" + "=" * 60)
print("SHAP EXPLAINABILITY")
print("=" * 60)
print("Computing SHAP values using XGBoost base model...")

xgb_model = stacked_model.named_estimators_["xgb"]
explainer = shap.TreeExplainer(xgb_model)
X_test_arr = np.array(X_test)
shap_values = explainer.shap_values(X_test_arr)

feature_names = list(X_test.columns)

# 9a. Summary plot — global feature importance
plt.figure()
shap.summary_plot(shap_values, X_test_arr, feature_names=feature_names,
                  show=False, plot_size=(10, 6))
plt.title("SHAP Summary Plot — Feature Importance")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.show()
print("SHAP summary plot saved as shap_summary.png")

# 9b. Bar plot — mean absolute SHAP values
plt.figure()
shap.summary_plot(shap_values, X_test_arr, feature_names=feature_names,
                  plot_type="bar", show=False, plot_size=(10, 6))
plt.title("SHAP Feature Importance (Mean |SHAP value|)")
plt.tight_layout()
plt.savefig("shap_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("SHAP importance plot saved as shap_importance.png")

# 9c. Waterfall plot for one prediction — explain a single customer
print("\nExplaining prediction for customer index 0 (test set):")
shap_exp = shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test_arr[0],
    feature_names=feature_names
)
plt.figure()
shap.waterfall_plot(shap_exp, show=False)
plt.tight_layout()
plt.savefig("shap_waterfall_customer0.png", dpi=150, bbox_inches="tight")
plt.show()
print("SHAP waterfall plot saved as shap_waterfall_customer0.png")


from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(y_test, y_pred_tuned)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Not Churned", "Churned"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix (Threshold = {best_threshold:.2f})")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()


# =============================================================================
# 10. FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("FILES GENERATED")
print("=" * 60)
print("  eda_plots.png")
print("  correlation_matrix.png")
print("  threshold_tuning.png")
print("  roc_curve.png")
print("  shap_summary.png")
print("  shap_importance.png")
print("  shap_waterfall_customer0.png")
print("\nDone!")
