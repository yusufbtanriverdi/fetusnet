import torch
import matplotlib.pyplot as plt

def plot_histograms_and_stats(outputs: torch.Tensor, targets: torch.Tensor):
    """
    Plots histograms of the outputs and targets, and prints their min, max, and mean values.

    Args:
        outputs (torch.Tensor): Predicted logits (unnormalized scores).
        targets (torch.Tensor): Ground truth probabilities (one-hot encoded or soft labels).
    """
    # Convert tensors to numpy for plotting
    outputs_np = outputs.detach().cpu().numpy().flatten()
    targets_np = targets.detach().cpu().numpy().flatten()

    # Print statistics
    print("Outputs - Min:", outputs_np.min(), "Max:", outputs_np.max(), "Mean:", outputs_np.mean(), "Std:", outputs_np.std(), "Sum:", outputs_np.sum())
    print("Targets - Min:", targets_np.min(), "Max:", targets_np.max(), "Mean:", targets_np.mean(), "Std:", targets_np.std(), "Sum:", targets_np.sum())

    # Plot histograms
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(outputs_np, bins=50, color='blue', alpha=0.7, label='Outputs')
    plt.title('Outputs Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(targets_np, bins=50, color='green', alpha=0.7, label='Targets')
    plt.title('Targets Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()

def imshow_target_distance_matrices(*targets: torch.Tensor, titles: list = None):
    """
    Visualize multiple target distance matrices using matplotlib with an interactive slider.

    Args:
        *targets (torch.Tensor): Variable number of ground truth distance matrices (assumed to be 3D).
    """
    # Convert tensors to numpy arrays for visualization
    targets_np = [target.cpu().detach().numpy()[0, 0] for target in targets]

    import matplotlib.widgets as widgets

    # Define a function to update the slices being displayed
    def update_slice(val):
        slice_idx = int(slider.val)
        for i, (ax, target_np) in enumerate(zip(axes, targets_np)):
            slice_np = target_np[slice_idx]
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
        slice_np = target_np[initial_slice]
        im = ax.imshow(slice_np, cmap='viridis', interpolation='nearest')
        title = titles[i] if titles and i < len(titles) else f'Target {i + 1}'
        ax.set_title(f'{title} (Slice {initial_slice})')
        plt.colorbar(im, ax=ax, label='Distance')

    # Add a slider for selecting slices
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = widgets.Slider(ax_slider, 'Slice', 0, targets_np[0].shape[0] - 1, valinit=initial_slice, valstep=1)

    # Connect the slider to the update function
    slider.on_changed(update_slice)

    plt.show()

class CombinedLoss(torch.nn.Module):
    def __init__(self, loss_fns, weights=None):
        """
        Initialize the combined loss function.

        Args:
            loss_fns: A list of loss function instances.
            weights: A list of weights for each loss function. If None, all weights are set to 1.
        """
        super(CombinedLoss, self).__init__()
        self.loss_fns = loss_fns
        self.weights = weights if weights is not None else [1.0] * len(loss_fns)

    def __str__(self):
        return super().__str__() + f"({self.loss_fns})"
    
    def forward(self, *args, **kwargs):
        """
        Compute the combined loss.

        Args:
            *args, **kwargs: Arguments to be passed to each loss function.

        Returns:
            The weighted sum of all loss functions.
        """
        total_loss = 0.0
        for weight, loss_fn in zip(self.weights, self.loss_fns):
            total_loss += weight * loss_fn(*args, **kwargs)
        return total_loss
    
