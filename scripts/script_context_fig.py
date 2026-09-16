"""
Conceptual point-region feedback-loop illustration for anatomical landmark refinement.

Run:
    python anatomical_context_refinement.py

Outputs:
    anatomical_context_refinement.png
    anatomical_context_refinement.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.lines import Line2D
import seaborn as sns


np.random.seed(7)

# ---------------------------------------------------------------------
# Compact-figure visual configuration
# ---------------------------------------------------------------------
figureWidth = 4.0
figureHeight = 4.0

titleFontSize = 7.2
annotationFontSize = 5.6
legendFontSize = 5.6

axisLineWidth = 0.38
contourLineWidth = 0.70
contextLineWidth = 0.92
annotationLineWidth = 0.58
updateArrowLineWidth = 0.92

groundTruthMarkerSize = 28
predictedMarkerSize = 20
groundTruthMarkerLineWidth = 1.25
predictedMarkerLineWidth = 0.70

legendGroundTruthMarkerSize = 5.5
legendPredictedMarkerSize = 4.6
legendLineWidth = 0.95

sns.set_style(
    'whitegrid',
    {
        'font.family': 'sans-serif',
        'font.sans-serif': 'Verdana'
    }
)
sns.set_theme(
    style='whitegrid',
    context='paper',
    font_scale=0.72,
    palette='husl'
)

# ---------------------------------------------------------------------
# Shared 2D coordinate domain
# ---------------------------------------------------------------------
x = np.linspace(-4.0, 4.0, 500)
y = np.linspace(-3.0, 3.0, 420)
X, Y = np.meshgrid(x, y)

groundTruthPoint = np.array([0.15, 0.15])
initialPrediction = np.array([1.05, 0.45])
refinedPrediction = np.array([0.38, 0.20])


def gaussian(center, sigma, amplitude=1.0):
    """Create a 2D isotropic Gaussian heatmap."""
    deltaX = X - center[0]
    deltaY = Y - center[1]

    return amplitude * np.exp(
        -(deltaX**2 + deltaY**2) / (2 * sigma**2)
    )


def irregularRegion(center, radiusX, radiusY, phase=0.0, pointCount=240):
    """Create an irregular closed region path for anatomical context."""
    theta = np.linspace(0, 2 * np.pi, pointCount, endpoint=True)

    wobble = (
        1
        + 0.12 * np.sin(3 * theta + phase)
        + 0.06 * np.sin(7 * theta - 0.5 * phase)
    )

    pointX = center[0] + radiusX * wobble * np.cos(theta)
    pointY = center[1] + radiusY * wobble * np.sin(theta)

    vertices = np.column_stack([pointX, pointY])
    codes = np.full(pointCount, Path.LINETO)
    codes[0] = Path.MOVETO

    return Path(vertices, codes)


def contextField(center, radiusX, radiusY, phase=0.0):
    """Create a softly varying elliptical anatomical support field."""
    normalizedX = (X - center[0]) / radiusX
    normalizedY = (Y - center[1]) / radiusY

    angle = np.arctan2(normalizedY, normalizedX)
    radialDistance = np.sqrt(normalizedX**2 + normalizedY**2)

    boundary = (
        1
        + 0.12 * np.sin(3 * angle + phase)
        + 0.06 * np.sin(7 * angle - 0.5 * phase)
    )

    return np.exp(-0.5 * (radialDistance / boundary)**4)


def styleAxis(axis, title):
    """Apply the shared compact styling to one panel."""
    axis.set_xlim(-4, 4)
    axis.set_ylim(-3, 3)
    axis.set_aspect('equal')

    axis.set_title(
        title,
        loc='left',
        fontsize=titleFontSize,
        fontweight='bold',
        pad=3.5
    )

    axis.set_xticks([])
    axis.set_yticks([])

    for spine in axis.spines.values():
        spine.set_visible(False)

    axis.axhline(
        0,
        color='#d0d5db',
        linewidth=axisLineWidth,
        zorder=0
    )
    axis.axvline(
        0,
        color='#d0d5db',
        linewidth=axisLineWidth,
        zorder=0
    )


def addLandmarkMarker(axis, point, markerType, label=None, zOrder=8):
    """Draw either a ground-truth or predicted landmark marker."""
    if markerType == 'groundTruth':
        axis.scatter(
            *point,
            marker='x',
            s=groundTruthMarkerSize,
            linewidths=groundTruthMarkerLineWidth,
            color='black',
            zorder=zOrder,
            label=label
        )
    else:
        axis.scatter(
            *point,
            marker='D',
            s=predictedMarkerSize,
            edgecolors='black',
            linewidths=predictedMarkerLineWidth,
            facecolors='#2c7fb8',
            zorder=zOrder,
            label=label
        )


# ---------------------------------------------------------------------
# Create compact multi-panel figure
# ---------------------------------------------------------------------
figure, axes = plt.subplots(
    2,
    2,
    figsize=(figureWidth, figureHeight),
    constrained_layout=True
)
axes = axes.ravel()

# ---------------------------------------------------------------------
# Panel 1: Initial target and predicted heatmaps
# ---------------------------------------------------------------------
axis = axes[0]
styleAxis(axis, '1. Initial landmark evidence')

targetHeatmap = gaussian(groundTruthPoint, 0.34)
predictedHeatmap = gaussian(initialPrediction, 0.72)

axis.contourf(
    X,
    Y,
    predictedHeatmap,
    levels=np.linspace(0.10, predictedHeatmap.max(), 7),
    cmap='Blues',
    alpha=0.72
)
axis.contour(
    X,
    Y,
    targetHeatmap,
    levels=[0.15, 0.35, 0.60, 0.82],
    colors='#d7301f',
    linewidths=contourLineWidth
)

addLandmarkMarker(
    axis,
    groundTruthPoint,
    'groundTruth',
    'Ground truth landmark'
)
addLandmarkMarker(
    axis,
    initialPrediction,
    'prediction',
    'Predicted landmark'
)

axis.text(
    -3.72,
    -2.58,
    'Target heatmap (peaked)',
    color='#b2182b',
    fontsize=annotationFontSize
)
axis.text(
    -3.72,
    -3.0,
    'Predicted heatmap (broader)',
    color='#2166ac',
    fontsize=annotationFontSize
)

# ---------------------------------------------------------------------
# Panel 2: Anatomical context initialization
# ---------------------------------------------------------------------
axis = axes[1]
styleAxis(axis, '2. Introduce anatomical context')

groundTruthContextPath = irregularRegion(
    groundTruthPoint + np.array([0.30, 0.0]),
    2.0,
    1.35,
    phase=0.4
)

atlasCenter = (
    0.58 * groundTruthPoint
    + 0.42 * initialPrediction
    + np.array([0.0, -0.08])
)

atlasContextPath = irregularRegion(
    atlasCenter,
    1.62,
    1.06,
    phase=1.1
)

axis.add_patch(
    PathPatch(
        groundTruthContextPath,
        facecolor='#ef8a62',
        edgecolor='#b2182b',
        alpha=0.18,
        linewidth=contextLineWidth,
        linestyle=(0, (3, 2)),
        zorder=2
    )
)
axis.add_patch(
    PathPatch(
        atlasContextPath,
        facecolor='#67a9cf',
        edgecolor='#2166ac',
        alpha=0.26,
        linewidth=contextLineWidth,
        zorder=3
    )
)

addLandmarkMarker(axis, groundTruthPoint, 'groundTruth')
addLandmarkMarker(axis, initialPrediction, 'prediction')

axis.annotate(
    'Unknown ground truth\nanatomical context',
    xy=(-1.72, 1.00),
    xytext=(-3.76, 2.10),
    fontsize=annotationFontSize,
    color='#9c1c13',
    arrowprops={
        'arrowstyle': '-',
        'color': '#9c1c13',
        'linewidth': annotationLineWidth
    }
)

axis.annotate(
    'Atlas-initialized,\nlandmark-adjusted context',
    xy=(1.68, -0.26),
    xytext=(1.34, -2.34),
    fontsize=annotationFontSize,
    color='#145a86',
    arrowprops={
        'arrowstyle': '-',
        'color': '#145a86',
        'linewidth': annotationLineWidth
    }
)

# ---------------------------------------------------------------------
# Panel 3: Context-guided landmark refinement
# ---------------------------------------------------------------------
axis = axes[2]
styleAxis(axis, '3. Context-guided\nlandmark update')

initialContextField = contextField(
    atlasCenter,
    1.62,
    1.06,
    phase=1.1
)

axis.contourf(
    X,
    Y,
    initialContextField,
    levels=np.linspace(0.18, 1, 7),
    cmap='Blues',
    alpha=0.56
)
axis.contour(
    X,
    Y,
    initialContextField,
    levels=[0.30, 0.58],
    colors='#2166ac',
    linewidths=contourLineWidth
)

combinedEvidence = (
    gaussian(initialPrediction, 0.72)
    * (0.20 + 0.80 * initialContextField)
)

axis.contour(
    combinedEvidence,
    levels=[
        combinedEvidence.max() * 0.22,
        combinedEvidence.max() * 0.48,
        combinedEvidence.max() * 0.75
    ],
    colors='#6a3d9a',
    linewidths=contourLineWidth
)

addLandmarkMarker(axis, groundTruthPoint, 'groundTruth')
addLandmarkMarker(axis, initialPrediction, 'prediction', zOrder=8)
addLandmarkMarker(axis, refinedPrediction, 'prediction', zOrder=9)

axis.annotate(
    '',
    xy=refinedPrediction,
    xytext=initialPrediction,
    arrowprops={
        'arrowstyle': '->',
        'color': '#1f4e79',
        'linewidth': updateArrowLineWidth,
        'mutation_scale': 7
    }
)

axis.text(
    -3.68,
    -2.56,
    'Updated predicted landmark\nusing an attention mechanism',
    fontsize=annotationFontSize,
    color='#1f4e79'
)

# ---------------------------------------------------------------------
# Panel 4: Landmark-guided context refinement
# ---------------------------------------------------------------------
axis = axes[3]
styleAxis(axis, '4. Landmark-guided\ncontext update')

updatedContextCenter = (
    0.76 * groundTruthPoint
    + 0.24 * refinedPrediction
    + np.array([0.10, 0.00])
)

previousContextPath = irregularRegion(
    atlasCenter,
    1.62,
    1.06,
    phase=1.1
)
updatedContextPath = irregularRegion(
    updatedContextCenter,
    1.45,
    0.95,
    phase=0.72
)

axis.add_patch(
    PathPatch(
        previousContextPath,
        facecolor='none',
        edgecolor='#7f8c8d',
        alpha=0.8,
        linewidth=contourLineWidth,
        linestyle=(0, (2, 2)),
        zorder=2
    )
)

axis.add_patch(
    PathPatch(
        updatedContextPath,
        facecolor='#41ae76',
        edgecolor='#006d2c',
        alpha=0.30,
        linewidth=contextLineWidth,
        zorder=4
    )
)

addLandmarkMarker(axis, groundTruthPoint, 'groundTruth')
addLandmarkMarker(axis, refinedPrediction, 'prediction')

axis.annotate(
    'Updated context conditioned\non the refined landmark',
    xy=(1.38, 0.48),
    xytext=(1.18, -2.28),
    fontsize=annotationFontSize,
    color='#006d2c',
    arrowprops={
        'arrowstyle': '-',
        'color': '#006d2c',
        'linewidth': annotationLineWidth
    }
)

axis.annotate(
    'Previous context',
    xy=(-0.90, 1.12),
    xytext=(-3.72, 2.10),
    fontsize=annotationFontSize,
    color='#59656a',
    arrowprops={
        'arrowstyle': '-',
        'color': '#59656a',
        'linewidth': annotationLineWidth
    }
)

# ---------------------------------------------------------------------
# Compact shared legend
# ---------------------------------------------------------------------
legendHandles = [
    Line2D(
        [0],
        [0],
        marker='x',
        color='black',
        linestyle='None',
        markersize=legendGroundTruthMarkerSize,
        markeredgewidth=1.25,
        label='Ground-truth landmark'
    ),
    Line2D(
        [0],
        [0],
        marker='D',
        color='black',
        markerfacecolor='#2c7fb8',
        linestyle='None',
        markersize=legendPredictedMarkerSize,
        markeredgewidth=0.70,
        label='Predicted landmark'
    ),
    Line2D(
        [0],
        [0],
        color='#b2182b',
        linewidth=legendLineWidth,
        linestyle=(0, (3, 2)),
        label='Unknown ground-truth context'
    ),
    Line2D(
        [0],
        [0],
        color='#2166ac',
        linewidth=legendLineWidth,
        label='Predicted anatomical context'
    )
]

figure.legend(
    handles=legendHandles,
    loc='lower center',
    ncol=2,
    bbox_to_anchor=(0.5, -0.035),
    frameon=False,
    fontsize=legendFontSize,
    handlelength=1.25,
    handletextpad=0.35,
    columnspacing=0.85,
    labelspacing=0.30
)

# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------
figure.savefig(
    'anatomical_context_refinement.png',
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.03
)
figure.savefig(
    'anatomical_context_refinement.pdf',
    bbox_inches='tight',
    pad_inches=0.03
)