import pandas as pd

df = pd.read_csv(r"..\cleaned_hotel_bookings.csv")
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

model = LogisticRegression()
model.fit(X_train,y_train)

pred = model.predict(X_test)

print(
    accuracy_score(y_test,pred)
)

print(
    classification_report(y_test,pred)
)