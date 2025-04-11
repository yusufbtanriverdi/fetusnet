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

def imshow_target_distance_matrices(target1: torch.Tensor, target2: torch.Tensor, target3: torch.Tensor):
    """
    Visualize three target distance matrices using matplotlib with an interactive slider.

    Args:
        target1 (torch.Tensor): First ground truth distance matrix (assumed to be 3D).
        target2 (torch.Tensor): Second ground truth distance matrix (assumed to be 3D).
        target3 (torch.Tensor): Third ground truth distance matrix (assumed to be 3D).
    """
    # Convert tensors to numpy arrays for visualization
    target1_np = target1.cpu().detach().numpy()[0, 0]
    target2_np = target2.cpu().detach().numpy()[0, 0]
    target3_np = target3.cpu().detach().numpy()[0, 0]

    import matplotlib.widgets as widgets

    # Define a function to update the slices being displayed
    def update_slice(val):
        slice_idx = int(slider.val)
        slice1_np = target1_np[slice_idx]
        slice2_np = target2_np[slice_idx]
        slice3_np = target3_np[slice_idx]

        ax1.imshow(slice1_np, cmap='viridis', interpolation='nearest')
        ax1.set_title(f'Target 1 (Slice {slice_idx})')

        ax2.imshow(slice2_np, cmap='viridis', interpolation='nearest')
        ax2.set_title(f'Target 2 (Slice {slice_idx})')

        ax3.imshow(slice3_np, cmap='viridis', interpolation='nearest')
        ax3.set_title(f'Target 3 (Slice {slice_idx})')

        fig.canvas.draw_idle()

    # Create a figure and axes for subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.2)

    # Initial slice to display
    initial_slice = 0
    slice1_np = target1_np[initial_slice]
    slice2_np = target2_np[initial_slice]
    slice3_np = target3_np[initial_slice]

    im1 = ax1.imshow(slice1_np, cmap='viridis', interpolation='nearest')
    ax1.set_title(f'Target 1 (Slice {initial_slice})')
    plt.colorbar(im1, ax=ax1, label='Distance')

    im2 = ax2.imshow(slice2_np, cmap='viridis', interpolation='nearest')
    ax2.set_title(f'Target 2 (Slice {initial_slice})')
    plt.colorbar(im2, ax=ax2, label='Distance')

    im3 = ax3.imshow(slice3_np, cmap='viridis', interpolation='nearest')
    ax3.set_title(f'Target 3 (Slice {initial_slice})')
    plt.colorbar(im3, ax=ax3, label='Distance')

    # Add a slider for selecting slices
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = widgets.Slider(ax_slider, 'Slice', 0, target1_np.shape[0] - 1, valinit=initial_slice, valstep=1)

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
    
