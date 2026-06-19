import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid', {'font.family':'sans-serif', 'font.sans-serif': 'Verdana'})
sns.set_theme('paper', 'whitegrid', font_scale=1.25, palette='husl')

df = pd.read_csv('assets/lambdas.csv')

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    data=df.iloc[:, 3:],
    cbar=True,
    cmap='GnBu',
    yticklabels=df['lambdas'],
    ax=ax,
)

ax.set_ylabel("Lambdas")
ax.set_xlabel("Landmarks")

plt.xticks(ha='right')
plt.tight_layout()
plt.show()  
fig.savefig('assets/lambdas.svg', format='svg')


