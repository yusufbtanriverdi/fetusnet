import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from matplotlib.patches import Circle

def plot_heatmaps_slices_from_coord(heatmaps, coord_tensor, argmax_tensor, titles=None):
    """
    Plot 2D heatmaps and 3D surfaces for slices along each axis.
    Columns = axes (0,1,2), Rows = 2D heatmap / 3D surface.

    Args:
        heatmaps (list of np.ndarray): List of 3D arrays (D, H, W) to plot.
        coord_tensor (tuple or list): Coordinate (x, y, z) to select slices.
        titles (list of str, optional): Titles for each heatmap.
        figsize (tuple): Figure size.
    """
    heatmaps = [t.detach().cpu().numpy() for t in heatmaps]
    n = len(heatmaps)
    axes = [0, 1, 2]  # Columns = axes
    coord_idx = tuple(int(round(c)) for c in coord_tensor)
    argmax_idx = tuple(int(round(c)) for c in argmax_tensor)

    if titles is None:
        titles = [f"Heatmap {i+1}" for i in range(n)]

    for i, heatmap in enumerate(heatmaps):
        fig, axs = plt.subplots(2, len(axes), figsize=(4*len(axes), 8),
                                subplot_kw={'projection': None})

        for j, ax in enumerate(axes):
            # Select slice along current axis
            ax2d = axs[0, j]
            if ax == 0:
                slice_data = heatmap[coord_idx[0], :, :]
                patch1 = Circle((coord_idx[2], coord_idx[1]), radius=1, color='white')
                patch2 = Circle((argmax_idx[2], argmax_idx[1]), radius=1, color='black')
                ax2d.add_patch(patch1)
                ax2d.add_patch(patch2)
            elif ax == 1:
                slice_data = heatmap[:, coord_idx[1], :]                
                patch1 = Circle((coord_idx[2], coord_idx[0]), radius=1, color='white')
                patch2 = Circle((argmax_idx[2], argmax_idx[0]), radius=1, color='black')
                ax2d.add_patch(patch1)
                ax2d.add_patch(patch2)
            else:
                slice_data = heatmap[:, :, coord_idx[2]]
                patch1 = Circle((coord_idx[1], coord_idx[0]), radius=1, color='white')
                patch2 = Circle((argmax_idx[1], argmax_idx[0]), radius=1, color='black')
                ax2d.add_patch(patch1)
                ax2d.add_patch(patch2)
            
            # 2D heatmap (top row)
            im = ax2d.imshow(slice_data, cmap='jet', origin='lower')
            ax2d.set_title(f"{titles[i]} - Axis {ax}")
            ax2d.set_axis_off()
            fig.colorbar(im, ax=ax2d, fraction=0.046, pad=0.04)

            # 3D surface (bottom row)
            ax__ = axs[1, j]
            ax__.set_axis_off()

            ax3d = fig.add_subplot(2, len(axes), len(axes)+j+1, projection='3d')
            H, W = slice_data.shape
            X, Y = np.meshgrid(np.arange(W), np.arange(H))
            # ax3d.set_axis_off()
            ax3d.plot_surface(X, Y, slice_data, cmap='jet', edgecolor='k', linewidth=0.2)
            ax3d.set_title(f"{titles[i]} Surface - Axis {ax}")

        plt.tight_layout()
        
    plt.show()
    plt.close()  # Free memory

if __name__ == "__main__":
    # Dummy 3D heatmaps
    heatmap1 = np.exp(-((np.indices((30, 50, 50))[1]-25)**2 + (np.indices((30, 50, 50))[2]-25)**2)/50)
    heatmap2 = np.exp(-((np.indices((30, 50, 50))[1]-25)**2 + (np.indices((30, 50, 50))[2]-25)**2)/20)

    coord = (15, 25, 25)

    # Plot each heatmap in separate figure with columns = axes, rows = image/surface
    plot_heatmaps_slices_from_coord([heatmap1, heatmap2], coord, titles=["Predicted", "Modulated"])
