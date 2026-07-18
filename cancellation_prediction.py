import pandas as pd

df = pd.read_csv(r"..\cleaned_hotel_bookings.csv")

print(df.columns)
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']

# Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

features = [
    'lead_time',
    'adr',
    'total_nights'
]

X = df[features]
y = df['is_canceled']

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train,y_train)

pred = log_model.predict(X_test)

print(accuracy_score(y_test,pred))
print(classification_report(y_test,pred))

# Feature engineering
df['is_family'] = ((df['children'] + df['babies']) > 0).astype(int)
df['lead_time_bucket'] = pd.cut(df['lead_time'],
                                bins=[0, 7, 30, 90, 365, df['lead_time'].max()],
                                labels=['last_minute', 'short', 'medium', 'long', 'very_long'])
df['room_mismatch'] = (df['reserved_room_type'] != df['assigned_room_type']).astype(int)

# RandomForest
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

importances = rf_model.feature_importances_
feature_importance = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
print(feature_importance)

from sklearn.inspection import permutation_importance
result = permutation_importance(rf_model, X_test, y_test, n_repeats=10, random_state=42)
importance = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)
print(importance)

from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Logistic Regression metrics
log_pred = pred
log_prob = log_model.predict_proba(X_test)[:,1]
log_results = {
    "Accuracy": accuracy_score(y_test, log_pred),
    "Precision": precision_score(y_test, log_pred),
    "Recall": recall_score(y_test, log_pred),
    "F1": f1_score(y_test, log_pred),
    "ROC-AUC": roc_auc_score(y_test, log_prob)
}

# RandomForest metrics
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:,1]
rf_results = {
    "Accuracy": accuracy_score(y_test, rf_pred),
    "Precision": precision_score(y_test, rf_pred),
    "Recall": recall_score(y_test, rf_pred),
    "F1": f1_score(y_test, rf_pred),
    "ROC-AUC": roc_auc_score(y_test, rf_prob)
}

results_df = pd.DataFrame([log_results, rf_results], index=["Logistic Regression", "Random Forest"])
print(results_df)

# --- Polished ROC Curve ---
fpr, tpr, thresholds = roc_curve(y_test, rf_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Random Forest (AUC = {roc_auc_score(y_test, rf_prob):.2f})")
plt.plot([0,1],[0,1], color="navy", lw=2, linestyle="--", label="Chance")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve - Booking Cancellation Prediction", fontsize=14, fontweight="bold")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")  # Save high-res PNG
plt.show()

# --- Polished Feature Importance Plot ---
plt.figure(figsize=(8,6))
feature_importance.head(15).plot(kind='barh', color="steelblue", edgecolor="black")
plt.title("Top 15 Feature Importances - Random Forest", fontsize=14, fontweight="bold")
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")  # Save high-res PNG
plt.show()
