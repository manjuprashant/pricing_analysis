import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

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

# 7. Feature importance
fi_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf.feature_importances_
}).sort_values(by="importance", ascending=False)

# 8. Save to CSV
output_file = "feature_importance.csv"
fi_df.to_csv(output_file, index=False)

print(f"✅ Feature importance exported to {output_file}")
