"""Conceptual point--region feedback-loop illustration for anatomical landmark refinement.
Run: python anatomical_context_refinement.py
Outputs: anatomical_context_refinement.png and anatomical_context_refinement.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.lines import Line2D

np.random.seed(7)

# Shared 2D coordinate domain
x = np.linspace(-4.0, 4.0, 500)
y = np.linspace(-3.0, 3.0, 420)
X, Y = np.meshgrid(x, y)
gt = np.array([0.15, 0.15])
pred0 = np.array([1.05, 0.45])
pred1 = np.array([0.38, 0.20])


def gaussian(mu, sigma, amplitude=1.0):
    dx = X - mu[0]
    dy = Y - mu[1]
    return amplitude * np.exp(-(dx**2 + dy**2) / (2 * sigma**2))


def irregular_region(center, rx, ry, phase=0.0, n=240):
    theta = np.linspace(0, 2*np.pi, n, endpoint=True)
    wobble = 1 + 0.12*np.sin(3*theta + phase) + 0.06*np.sin(7*theta - 0.5*phase)
    px = center[0] + rx * wobble * np.cos(theta)
    py = center[1] + ry * wobble * np.sin(theta)
    verts = np.column_stack([px, py])
    codes = np.full(n, Path.LINETO)
    codes[0] = Path.MOVETO
    return Path(verts, codes)


def context_field(center, rx, ry, phase=0.0):
    # Elliptical, softly varying support field used only for visualization.
    dx = (X - center[0]) / rx
    dy = (Y - center[1]) / ry
    angle = np.arctan2(dy, dx)
    radial = np.sqrt(dx**2 + dy**2)
    boundary = 1 + 0.12*np.sin(3*angle + phase) + 0.06*np.sin(7*angle - 0.5*phase)
    return np.exp(-0.5 * (radial / boundary)**4)


def style(ax, title):
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title(title, loc='left', fontsize=12, weight='bold', pad=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(0, color='#d0d5db', lw=0.7, zorder=0)
    ax.axvline(0, color='#d0d5db', lw=0.7, zorder=0)


def marker(ax, point, kind, label=None, z=8):
    if kind == 'gt':
        ax.scatter(*point, marker='x', s=90, linewidths=2.7, c='black', zorder=z, label=label)
    else:
        ax.scatter(*point, marker='D', s=58, edgecolors='black', linewidths=1.2,
                   facecolors='#2c7fb8', zorder=z, label=label)

fig, axes = plt.subplots(1, 4, figsize=(16.8, 4.6), constrained_layout=True)

# Panel 1: Target and predicted heatmaps.
ax = axes[0]
style(ax, '1. Initial landmark evidence')
target = gaussian(gt, 0.34)
predicted = gaussian(pred0, 0.72)
ax.contourf(X, Y, predicted, levels=np.linspace(0.10, predicted.max(), 7), cmap='Blues', alpha=0.72)
ax.contour(X, Y, target, levels=[0.15, 0.35, 0.60, 0.82], colors=['#d7301f'], linewidths=1.25)
marker(ax, gt, 'gt', 'Ground truth landmark')
marker(ax, pred0, 'pred', 'Predicted landmark')
ax.text(-3.75, -2.62, 'Target heatmap: peaked', color='#b2182b', fontsize=9)
ax.text(-3.75, -2.89, 'Predicted heatmap: broader', color='#2166ac', fontsize=9)

# Panel 2: Unknown ground truth context plus atlas-initialized/landmark-adjusted context.
ax = axes[1]
style(ax, '2. Introduce anatomical context')
gt_path = irregular_region(gt + np.array([0.30, 0.0]), 2.0, 1.35, phase=0.4)
atlas_center = 0.58 * gt + 0.42 * pred0 + np.array([0.0, -0.08])
atlas_path = irregular_region(atlas_center, 1.62, 1.06, phase=1.1)
ax.add_patch(PathPatch(gt_path, facecolor='#ef8a62', edgecolor='#b2182b', alpha=0.18,
                       lw=1.6, linestyle=(0, (4, 3)), zorder=2))
ax.add_patch(PathPatch(atlas_path, facecolor='#67a9cf', edgecolor='#2166ac', alpha=0.26,
                       lw=1.8, zorder=3))
marker(ax, gt, 'gt')
marker(ax, pred0, 'pred')
ax.annotate('Unknown ground-truth\nanatomical context', xy=(-1.72, 1.00), xytext=(-3.82, 2.22),
            fontsize=8.7, color='#9c1c13', arrowprops=dict(arrowstyle='-', color='#9c1c13', lw=0.9))
ax.annotate('Atlas-initialized,\nlandmark-adjusted context', xy=(1.78, -0.30), xytext=(1.48, -2.44),
            fontsize=8.7, color='#145a86', arrowprops=dict(arrowstyle='-', color='#145a86', lw=0.9))

# Panel 3: Context guides landmark update.
ax = axes[2]
style(ax, '3. Context-guided landmark update')
ctx0 = context_field(atlas_center, 1.62, 1.06, phase=1.1)
ax.contourf(X, Y, ctx0, levels=np.linspace(0.18, 1, 7), cmap='Blues', alpha=0.56)
ax.contour(X, Y, ctx0, levels=[0.30, 0.58], colors='#2166ac', linewidths=1.2)
combined = gaussian(pred0, 0.72) * (0.20 + 0.80 * ctx0)
ax.contour(combined, levels=[combined.max()*0.22, combined.max()*0.48, combined.max()*0.75],
           colors='#6a3d9a', linewidths=1.3)
marker(ax, gt, 'gt')
marker(ax, pred0, 'pred', z=8)
marker(ax, pred1, 'pred', z=9)
ax.annotate('', xy=pred1, xytext=pred0, arrowprops=dict(arrowstyle='->', color='#1f4e79', lw=1.6))
ax.text(-3.72, -2.62, r'$	ilde{f l}^{(1)}$: updated predicted landmark', fontsize=9, color='#1f4e79')
ax.text(-3.72, -2.89, r'Heatmap $	imes$ contextual support', fontsize=9, color='#5e3c99')

# Panel 4: Updated landmark guides context update.
ax = axes[3]
style(ax, '4. Landmark-guided context update')
ctx1_center = 0.76 * gt + 0.24 * pred1 + np.array([0.10, 0.00])
old_path = irregular_region(atlas_center, 1.62, 1.06, phase=1.1)
new_path = irregular_region(ctx1_center, 1.45, 0.95, phase=0.72)
ax.add_patch(PathPatch(old_path, facecolor='none', edgecolor='#7f8c8d', alpha=0.8, lw=1.2,
                       linestyle=(0, (3, 3)), zorder=2))
ax.add_patch(PathPatch(new_path, facecolor='#41ae76', edgecolor='#006d2c', alpha=0.30,
                       lw=1.8, zorder=4))
marker(ax, gt, 'gt')
marker(ax, pred1, 'pred')
ax.annotate('Updated context\nconditioned on $\tilde{\bf l}^{(1)}$', xy=(1.45, 0.52), xytext=(1.32, -2.34),
            fontsize=8.7, color='#006d2c', arrowprops=dict(arrowstyle='-', color='#006d2c', lw=0.9))
ax.annotate('Previous context', xy=(-0.93, 1.17), xytext=(-3.78, 2.24),
            fontsize=8.7, color='#59656a', arrowprops=dict(arrowstyle='-', color='#59656a', lw=0.9))

handles = [
    Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=9, markeredgewidth=2.3,
           label='Ground-truth landmark'),
    Line2D([0], [0], marker='D', color='black', markerfacecolor='#2c7fb8', linestyle='None', markersize=7,
           label='Predicted landmark'),
    Line2D([0], [0], color='#b2182b', lw=1.5, linestyle=(0, (4, 3)), label='Unknown ground-truth context'),
    Line2D([0], [0], color='#2166ac', lw=2, label='Predicted anatomical context'),
]
fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.07), frameon=False, fontsize=9)
fig.suptitle('Alternating point--region refinement with atlas-informed anatomical context', fontsize=15, weight='bold', y=1.03)

fig.savefig('anatomical_context_refinement.png', dpi=300, bbox_inches='tight')
fig.savefig('anatomical_context_refinement.pdf', bbox_inches='tight')
