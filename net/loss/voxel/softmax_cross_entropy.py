import torch
import torch.nn as nn
from net.loss.utils import plot_histograms_and_stats
      
class SoftmaxCrossEntropyLoss(nn.Module):
    def __init__(self, reduction: str = 'mean', eps: float = 1e-20):
        """
        Custom Softmax Cross-Entropy Loss.

        Args:
            reduction (str): Specifies the reduction method to apply.
                - 'mean': Returns the mean of the loss.
                - 'sum': Returns the sum of the loss.
                - 'none': Returns the loss without reduction.
            eps (float): A small value to avoid numerical instability (e.g., log(0)).
        """
        super(SoftmaxCrossEntropyLoss, self).__init__()
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'."
        self.reduction = reduction
        self.eps = eps

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the Softmax Cross-Entropy Loss.

        Args:
            outputs (torch.Tensor): Predicted logits (unnormalized scores).
            targets (torch.Tensor): Ground truth probabilities (one-hot encoded or soft labels).

        Returns:
            torch.Tensor: Computed loss.
        """
        # Normalize the targets to ensure they sum to 1 (softmax-like behavior)
        targets = targets / (targets.sum(dim=-1, keepdim=True) + self.eps)

        # Apply softmax to the outputs to convert logits to probabilities
        outputs = nn.functional.softmax(outputs, dim=-1)

        # Call the function to visualize
        plot_histograms_and_stats(outputs, targets)
        # Compute the element-wise cross-entropy loss
        # Adding `self.eps` ensures numerical stability by avoiding log(0)
        loss = targets * torch.log(outputs + self.eps)

        # Sum the loss over spatial dimensions (e.g., D, H, W for 3D or H, W for 2D)
        loss = loss.view(loss.size(0), -1).sum(dim=-1)

        # Apply the specified reduction method
        if self.reduction == 'mean':
            # Return the mean loss over the batch
            return loss.mean()
        elif self.reduction == 'sum':
            # Return the sum of all loss values over the batch
            return loss.sum()
        elif self.reduction == 'none':
            # Return the per-sample loss without reduction
            return loss