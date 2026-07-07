import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
# Load dataset
df = pd.read_csv(r'archive\hotel_bookings.csv')
df.head()

contingency_table = pd.crosstab(df['market_segment'], df['is_canceled'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print('Chi-Square Statistic:', chi2)
print('P-Value:', p)
print('Degrees of Freedom:', dof)

contingency_table = pd.crosstab(df['customer_type'], df['is_canceled'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print('Chi-Square Statistic:', chi2)
print('P-Value:', p)
print('Degrees of Freedom:', dof)

cancelled = df[df['is_canceled'] == 1]['adr']
not_cancelled = df[df['is_canceled'] == 0]['adr']
t_stat, p = ttest_ind(cancelled, not_cancelled, nan_policy='omit')
print('T-Statistic:', t_stat)
print('P-Value:', p)

