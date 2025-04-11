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
    print("Outputs - Min:", outputs_np.min(), "Max:", outputs_np.max(), "Mean:", outputs_np.mean())
    print("Targets - Min:", targets_np.min(), "Max:", targets_np.max(), "Mean:", targets_np.mean())

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