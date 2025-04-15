import torch
import torch.nn as nn

class KullbackLeiblerDivLossV2(nn.Module):
    """
    Custom Kullback-Leibler (KLD) divergence loss implementation.
    This loss measures the divergence between two probability distributions.
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-20):
        """
        Initializes the Kullback-Leibler divergence loss.

        Args:
            reduction (str): Specifies the reduction method to apply.
                - 'mean': Returns the mean of the loss.
                - 'sum': Returns the sum of the loss.
                - 'none': Returns the loss without reduction.
            eps (float): A small value to avoid numerical instability (e.g., log(0)).
        """
        super(KullbackLeiblerDivLossV2, self).__init__()
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'."
        self.reduction = reduction
        self.eps = eps

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the Kullback-Leibler divergence loss.

        Args:
            outputs (torch.Tensor): Predicted values (logits).
            targets (torch.Tensor): Ground truth probability distributions.

        Returns:
            torch.Tensor: Computed loss.
        """
        # Normalize the targets to ensure they sum to 1 (softmax-like behavior)
        targets = targets / (targets.view(targets.size(0), -1).sum(dim=-1, keepdim=True))

        # Apply softmax to the flattened outputs
        outputs = torch.nn.functional.softmax(outputs.view(outputs.size(0), -1), dim=-1).view_as(outputs)

        # Compute element-wise KLD: targets * (log(targets) - log(outputs))
        loss = targets * (torch.log(targets) - torch.log(outputs))

        # Sum over spatial dimensions (e.g., D, H, W for 3D or H, W for 2D)
        loss = loss.view(loss.size(0), -1).sum(dim=-1)

        # Apply the specified reduction method
        if self.reduction == 'mean':
            return loss.mean()  # Return the mean loss over the batch
        elif self.reduction == 'sum':
            return loss.sum()  # Return the sum of all loss values
        elif self.reduction == 'none':
            return loss  # Return the per-sample loss without reduction