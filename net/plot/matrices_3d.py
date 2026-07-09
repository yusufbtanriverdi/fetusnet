import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import widgets
import imageio
from typing import List, Optional

def plot_3d_matrices(
    *matrices: torch.Tensor,
    titles: Optional[List[str]] = None,
    gif_path: Optional[str] = None,
    axis: int = 0,
    interactive: bool = True,
    fps: int = 2,
):
    """
    Visualize multiple 3D target distance matrices as slices, with optional GIF export.

    Args:
        *matrices (torch.Tensor): Variable number of 3D distance matrices (shape [D, H, W]).
        titles (list, optional): Titles for each target distance matrix.
        gif_path (str, optional): If provided, saves the visualization as a GIF.
        axis (int): Axis along which to slice (default=0, i.e., depth).
        interactive (bool): If True, use a slider to browse slices. If False, auto-generate a GIF.
        fps (int): Frames per second for GIF output.
    """
    matrices_np = [t.detach().cpu().numpy() for t in matrices]
    num_matrices = len(matrices_np)
    depth = matrices_np[0].shape[axis]

    # Ensure titles
    if titles is None:
        titles = [f"Target {i+1}" for i in range(num_matrices)]

    # Consistent color scale across all slices
    vmin = min(t.min() for t in matrices_np)
    vmax = max(t.max() for t in matrices_np)

    fig, axes = plt.subplots(1, num_matrices, figsize=(5 * num_matrices, 5))
    if num_matrices == 1:
        axes = [axes]

    def show_slice(slice_idx: int, cbar=False):
        for ax, target_np, title in zip(axes, matrices_np, titles):
            # Select slice along chosen axis
            slice_np = np.take(target_np, slice_idx, axis=axis)
            ax.clear()
            im = ax.imshow(slice_np, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"{title} (Slice {slice_idx})")
            # ax.axis("off")
            if cbar:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Distance")

    if interactive:
        # Initial slice
        show_slice(0, cbar=True)

        # Slider
        plt.subplots_adjust(bottom=0.2)
        ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor="lightgoldenrodyellow")
        slider = widgets.Slider(ax_slider, "Slice", 0, depth - 1, valinit=0, valstep=1)

        def update(val):
            show_slice(int(slider.val))
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()

    else:
        images = []
        for slice_idx in range(depth):
            show_slice(slice_idx)
            fig.canvas.draw()
            # Capture image
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(image)

        plt.close(fig)

        if gif_path:
            imageio.mimsave(gif_path, images, fps=fps)

# if __name__ == "__main__":
#     # Dummy 3D volumes
#     t1 = torch.rand(30, 64, 64)  # (D, H, W)
#     t2 = torch.rand(30, 64, 64) * 2  # scaled version

#     # Interactive browsing
#     plot_3d_matrices(t1, t2, titles=["Random A", "Random B"], interactive=True)

#     # Auto-GIF mode
#     plot_3d_matrices(t1, t2, titles=["Random A", "Random B"], gif_path="distances.gif", interactive=False, fps=5)
