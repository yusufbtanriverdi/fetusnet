import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Fake heatmap data
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-(X**2 + Y**2))          # Predicted heatmap
Z2 = np.exp(-(X**2 + Y**2)/0.5)      # Modulated heatmap

fig, axs = plt.subplots(2, 4, figsize=(12, 6),
                        subplot_kw={'projection': None})

# Example 1 - predicted heatmap
axs[0, 0].imshow(Z1, cmap='jet')
axs[0, 0].set_title("Predicted Heatmap")

ax3d = fig.add_subplot(2, 4, 2, projection='3d')
ax3d.plot_surface(X, Y, Z1, cmap='jet')

# Example 1 - modulated heatmap
ax3d2 = fig.add_subplot(2, 4, 3, projection='3d')
ax3d2.plot_surface(X, Y, Z2, cmap='jet')

# Example 2 (just repeat with different data if you want)

# Add labels
fig.text(0.25, 0.95, "Example 1", ha='center', fontsize=12)
fig.text(0.25, 0.45, "Example 2", ha='center', fontsize=12)

plt.tight_layout()
plt.show()

def plot_heatmaps(heatmaps, titles):
    fig, axs = plt.subplots(2, 4, figsize=(12, 6),
                            subplot_kw={'projection': None})
    
    pass