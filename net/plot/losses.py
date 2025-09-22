import torch
from matplotlib import widgets
import matplotlib.pyplot as plt

def imshow_target_distance_matrices_to_gif(*targets: torch.Tensor, titles: list = None, gif_path: str = "distance_matrices.gif"):
    """
    Visualize multiple target distance matrices and save the visualization as a GIF.

    Args:
        *targets (torch.Tensor): Variable number of ground truth distance matrices (assumed to be 3D).
        titles (list): Titles for each target distance matrix.
        gif_path (str): Path to save the generated GIF.
    """
    # Convert tensors to numpy arrays for visualization
    # targets_np = [target.cpu().detach().numpy()[0, 0] for target in targets] # Assuming targets have shape (1, 1, D, H, W)
    targets_np = [target.cpu().detach().numpy() for target in targets]
    # Define a function to update the slices being displayed
    def update_slice(val):
        slice_idx = int(slider.val)
        for i, (ax, target_np) in enumerate(zip(axes, targets_np)):
            slice_np = target_np[:, :, slice_idx]
            ax.imshow(slice_np, cmap='viridis', interpolation='nearest')
            title = titles[i] if titles and i < len(titles) else f'Target {i + 1}'
            ax.set_title(f'{title} (Slice {slice_idx})')

        fig.canvas.draw_idle()

    # Create a figure and axes for subplots
    num_targets = len(targets_np)
    fig, axes = plt.subplots(1, num_targets, figsize=(5 * num_targets, 5))
    if num_targets == 1:
        axes = [axes]  # Ensure axes is iterable for a single target
    plt.subplots_adjust(bottom=0.2)

    # Initial slice to display
    initial_slice = 0
    for i, (ax, target_np) in enumerate(zip(axes, targets_np)):
        slice_np = target_np[:, :, initial_slice]
        im = ax.imshow(slice_np, cmap='viridis', interpolation='nearest')
        title = titles[i] if titles and i < len(titles) else f'Target {i + 1}'
        ax.set_title(f'{title} (Slice {initial_slice})')
        ax.axis('off')  # Hide axes
        plt.colorbar(im, ax=ax, label='Distance')

    # Add a slider for selecting slices
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = widgets.Slider(ax_slider, 'Slice', 0, targets_np[0].shape[0] - 1, valinit=initial_slice, valstep=1)

    # Connect the slider to the update function
    slider.on_changed(update_slice)
    plt.show()
    
    # # Automatically slide through slices and save as GIF frames
    # images = []
    # for slice_idx in range(targets_np[0].shape[0]):
    #     slider.set_val(slice_idx)
    #     fig.canvas.draw_idle()  # Use draw_idle instead of draw for better performance
    #     plt.pause(0.1)  # Pause to allow visualization

    #     # Capture the current figure as an image
    #     image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    #     image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #     images.append(image)

    # plt.close(fig)  # Close the figure to release memory
    # # Save the collected images as a GIF
    # imageio.mimsave(gif_path, images, fps=1)
