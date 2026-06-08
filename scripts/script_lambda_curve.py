import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid', {'font.family':'sans-serif', 'font.sans-serif': 'Verdana'})
sns.set_theme('paper', 'whitegrid', font_scale=1.25, palette='husl')

df = pd.read_csv('assets/lambdas.csv')

# Use plt.subplots to easily grab both the figure and axis objects
fig, ax = plt.subplots(figsize=(6, 6))

sns.scatterplot(
    data=df,
    x='lambdas',
    y='dmean', 
    size='dmean',
    sizes=(100, 400),
    legend=None,
    hue='dmean',
    palette="dark:#5A9_r",
    ax=ax # Explicitly draw onto our captured axis
)

# 1. Annotate dmean values directly on top of each point
for _, row in df.iterrows():
    ax.annotate(
        f"{row['dmean']:.2f}",          # Formats value to 2 decimal places
        (row['lambdas'], row['dmean']), # Point coordinates
        textcoords="offset points",     # Relative positioning
        xytext=(0, 12),                 # Push text 12 points up to clear the large bubbles
        ha='center',                    # Center text horizontally
        va='bottom',                    # Align text from its bottom edge
        fontweight='semibold',
        color='#333333'
    )

# 2. Turn off Y-axis tick labels (numbers), but keep the physical tick lines
ax.tick_params(axis='y', labelleft=False)

# Optional: If you want to remove the main Y-axis title entirely, use plt.ylabel('')
# Otherwise, this keeps the title descriptor while the grid numbers stay hidden.
plt.ylabel('d-mean Score (mm)', labelpad=15)

plt.xticks(ha='right')
plt.xlabel('Coefficient (\u03BB) of EMD-regularization')
plt.tight_layout()

plt.show()  
fig.savefig('assets/lambdas.svg', format='svg')