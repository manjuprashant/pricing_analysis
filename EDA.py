import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Save charts without opening windows

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Load dataset

df = pd.read_csv(r"..\cleaned_hotel_bookings.csv")

# Dataset Information

print("Dataset Shape:", df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nSummary Statistics:")
print(df.describe())

# -------------------------

# Cancellation Distribution

# -------------------------

plt.figure(figsize=(6,4))
sns.countplot(x='is_canceled', data=df)
plt.title("Booking Cancellation Distribution")
plt.savefig("cancellation_distribution.png")
plt.close()

# -------------------------

# Lead Time Distribution

# -------------------------

plt.figure(figsize=(8,5))
sns.histplot(df['lead_time'], bins=50, kde=True)
plt.title("Lead Time Distribution")
plt.savefig("lead_time_distribution.png")
plt.close()

# -------------------------

# Lead Time vs Cancellation

# -------------------------

plt.figure(figsize=(8,5))
sns.boxplot(x='is_canceled', y='lead_time', data=df)
plt.title("Lead Time vs Cancellation")
plt.savefig("leadtime_vs_cancellation.png")
plt.close()

# -------------------------

# ADR vs Cancellation

# -------------------------

plt.figure(figsize=(8,5))
sns.boxplot(x='is_canceled', y='adr', data=df)
plt.title("ADR vs Cancellation")
plt.savefig("adr_vs_cancellation.png")
plt.close()

# -------------------------

# Correlation Matrix

# -------------------------

numeric_df = df.select_dtypes(include=['int64','float64'])

plt.figure(figsize=(14,10))
sns.heatmap(
numeric_df.corr(),
cmap='coolwarm',
annot=False
)
plt.title("Correlation Matrix")
plt.savefig("correlation_matrix.png")
plt.close()

# -------------------------

# Correlation with Cancellation

# -------------------------

correlation = numeric_df.corr()['is_canceled'].sort_values(
ascending=False
)

correlation.to_csv(
"cancellation_correlation.csv",
header=['correlation']
)

print("\nTop Correlations With Cancellation:")
print(correlation.head(10))

# -------------------------

# T-Test

# -------------------------

cancelled = df[df['is_canceled'] == 1]['lead_time']
not_cancelled = df[df['is_canceled'] == 0]['lead_time']

t_stat, p_value = ttest_ind(
cancelled,
not_cancelled
)

print("\nT-Test Results")
print("T Statistic:", t_stat)
print("P Value:", p_value)

# -------------------------

# Save Report

# -------------------------

# -------------------------
# Save Report
# -------------------------

with open("EDA_Report.txt", "w") as f:
    f.write("HOTEL BOOKINGS EDA REPORT\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Dataset Shape: {df.shape}\n\n")

    f.write("Data Types:\n")
    f.write(str(df.dtypes))
    f.write("\n\n")

    f.write("Summary Statistics:\n")
    f.write(str(df.describe()))
    f.write("\n\n")

    f.write("Top Correlations With Cancellation:\n")
    f.write(str(correlation.head(10)))
    f.write("\n\n")

    f.write("T-Test Results\n")
    f.write(f"T Statistic: {t_stat}\n")
    f.write(f"P Value: {p_value}\n")

print("EDA report saved successfully!")