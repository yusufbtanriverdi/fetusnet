import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style(
    'whitegrid',
    {'font.family': 'sans-serif', 'font.sans-serif': 'Verdana'}
)
sns.set_theme(
    'paper',
    'whitegrid',
    font_scale=1.25,
    palette='husl'
)

# Read file
df = pd.read_csv("C:/Users/user/Projeler/Ph.D/Research/source/sinMaternitat__fold0.csv")
df['week'] = (
    df['week']
    .astype(str)
    .str.replace('semanas', '', regex=False)
    .str.strip()
    .astype(int)
)

# Count unique weeks for each (npid, nsid)
week_counts = (
    df.groupby(['week'])['nsid']
      .nunique()
      .reset_index(name='n_unique_weeks')
)

print(week_counts)
plt.figure(figsize=(12, 6))

week_counts['label'] = (
    week_counts['week'].astype(str)
)

sns.barplot(
    data=week_counts,
    x='label',
    y='n_unique_weeks'
)

plt.xlabel('Gestational Ages (weeks)')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Gestational Age (weeks)', labelpad=15)
plt.ylabel('Count', labelpad=15)
plt.tight_layout()
plt.show()  