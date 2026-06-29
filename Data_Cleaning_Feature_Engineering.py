import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading dataset...")

# Load Dataset
df = pd.read_csv("../archive/hotel_bookings.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# ----------------------------
# HANDLE MISSING VALUES
# ----------------------------

df['agent'] = df['agent'].fillna(0)
df['company'] = df['company'].fillna(0)
df['children'] = df['children'].fillna(0)

# Remove remaining nulls if any
df.dropna(inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sum())

# ----------------------------
# REMOVE DUPLICATES
# ----------------------------

before = len(df)

df.drop_duplicates(inplace=True)

after = len(df)

print(f"\nDuplicates Removed: {before - after}")

# ----------------------------
# ADR OUTLIER TREATMENT
# ----------------------------

print("\nTreating ADR Outliers...")

Q1 = df['adr'].quantile(0.25)
Q3 = df['adr'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[
    (df['adr'] >= lower_bound)
    & (df['adr'] <= upper_bound)
]

print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------

print("\nCreating Features...")

df['total_stay'] = (
    df['stays_in_weekend_nights']
    + df['stays_in_week_nights']
)

df['total_guests'] = (
    df['adults']
    + df['children']
    + df['babies']
)

df['is_family'] = np.where(
    df['total_guests'] > 2,
    1,
    0
)

month_map = {
    'January':'Winter',
    'February':'Winter',
    'March':'Spring',
    'April':'Spring',
    'May':'Spring',
    'June':'Summer',
    'July':'Summer',
    'August':'Summer',
    'September':'Autumn',
    'October':'Autumn',
    'November':'Autumn',
    'December':'Winter'
}

df['booking_season'] = (
    df['arrival_date_month']
    .map(month_map)
)

# ----------------------------
# SAVE CLEANED DATASET
# ----------------------------

output_file = "../cleaned_hotel_bookings.csv"

df.to_csv(output_file, index=False)

print("\nCleaned Dataset Saved:")
print(output_file)

print("\nFinal Dataset Shape:")
print(df.shape)

# ----------------------------
# VISUALIZATION
# ----------------------------

plt.figure(figsize=(8,4))
sns.boxplot(x=df['adr'])
plt.title("ADR Distribution After Outlier Removal")
plt.show()

plt.figure(figsize=(8,4))
sns.histplot(df['adr'], kde=True)
plt.title("ADR Distribution")
plt.show()

print("\nWeek 1 Completed Successfully")