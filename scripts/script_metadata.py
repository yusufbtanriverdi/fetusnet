import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style(
    {'font.family': 'sans-serif', 'font.sans-serif': 'Verdana'}
)
sns.set_theme(
    'paper',
    font_scale=2,
    palette='husl'
)

cm = np.array([
    [27, 23],
    [1,  0],
    [0,  1]
])

labels1 = ["White/European", "North African", "Pakistani"]
labels2 = ["Female", "Male"]
fig, ax = plt.subplots(figsize=(5, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="coolwarm",
    xticklabels=labels2,
    yticklabels=labels1,
    cbar=False,
    linewidths=1,
    linecolor="white",
    square=True,
    ax=ax
)

# ax.set_xlabel("F")
# ax.set_ylabel("M")

plt.tight_layout()
plt.show()
fig.savefig('assets/metadata.png', format='png')
