import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style('whitegrid', {'font.family':'sans-serif', 'font.sans-serif': 'Verdana'})
sns.set_theme('paper', 'whitegrid', font_scale=1.25, palette='husl')

df = pd.read_csv('assets/lambdas.csv')
fig = plt.figure(figsize=(6, 6))

sns.scatterplot(
    data=df,
    x='lambdas',
    y='dmean', 
    size='dmean',
    sizes=(100, 400),
    legend=None,
    hue='dmean',
    palette="dark:#5A9_r"
)

plt.xticks(ha='right')
plt.xlabel('Coefficient (\u03BB) of EMD-regularization')
plt.ylabel('d-mean Score (mm)', labelpad=15)
plt.tight_layout()
plt.show()  
fig.savefig('assets/lambdas.svg', format='svg')

