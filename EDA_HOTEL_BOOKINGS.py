# Cancellation Distribution
import pandas as pd

df = pd.read_csv(r"..\archive\hotel_bookings.csv")
data = df

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(
    x='is_canceled',
    data=df
)

plt.title('Booking Cancellation Distribution')
plt.show()


#ADR DISTRIBUTION

sns.histplot(
    df['adr'],
    bins=50,
    kde=True
)

plt.title('Average Daily Rate Distribution')
plt.show()

# CORRELATION HEATMAP

plt.figure(figsize=(12,8))

sns.heatmap(
    df.corr(numeric_only=True),
    cmap='coolwarm'
)

plt.title('Correlation Matrix')
plt.show()

print("LEAD TIME VS CANCELLATION")

sns.boxplot(
    x='is_canceled',
    y='lead_time',
    data=df
)

plt.title('Lead Time vs Cancellation')
plt.show()

