import torch
import matplotlib.pyplot as plt

def plot_histograms_and_stats(outputs: torch.Tensor, targets: torch.Tensor, save_path = None):
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

    if save_path:
        plt.savefig(save_path + 'histograms_outputs_targets.png' )
    plt.close()  # Close the plot to free memory
