import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score

# 1. Dataset path
data_path = "cleaned_hotel_bookings.csv"

if not os.path.exists(data_path):
    print(f"ERROR: Dataset not found at {data_path}")
    exit()

# 2. Load dataset
df = pd.read_csv(data_path)

# 3. Target column check
if "is_canceled" not in df.columns:
    print("ERROR: 'is_canceled' column not found in dataset.")
    exit()

y = df["is_canceled"]
X = df.drop(columns=["is_canceled"])

# 4. Encode categorical columns
for col in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Fit RandomForest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# 7. ROC curve data
y_pred_proba = rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr})
roc_df.to_csv("roc_data.csv", index=False)

# 8. Accuracy metrics
acc = accuracy_score(y_test, rf.predict(X_test))
prec = precision_score(y_test, rf.predict(X_test))
rec = recall_score(y_test, rf.predict(X_test))

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "ROC AUC"],
    "Value": [acc, prec, rec, auc]
})
metrics_df.to_csv("model_metrics.csv", index=False)

roc_df = pd.DataFrame({
    "FPR": fpr,
    "TPR": tpr,
    "Baseline": fpr
}).sort_values(by="FPR")   # ensures ascending order

roc_df.to_csv("roc_data.csv", index=False)


roc_df = pd.DataFrame({
    "FPR": fpr,
    "TPR": tpr,
    "Baseline": fpr   # diagonal line
})
roc_df.to_csv("roc_data.csv", index=False)


print("✅ ROC data exported to roc_data.csv")
print("✅ Model metrics exported to model_metrics.csv")
print(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, ROC AUC: {auc:.3f}")
