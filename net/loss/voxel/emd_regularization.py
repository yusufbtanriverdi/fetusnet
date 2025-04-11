import torch
import torch.nn as nn

class DistanceMatrixLoss(nn.Module):
    """
    Custom Distance Matrix Loss.

    This loss function computes a custom loss based on the distance matrix 
    between predicted and ground truth values. It supports different reduction 
    methods: 'mean', 'sum', and 'none'.

    Args:
        reduction (str): Specifies the reduction method to apply.
            - 'mean': Returns the mean of the loss.
            - 'sum': Returns the sum of the loss.
            - 'none': Returns the loss without reduction.
        lambda_ (float): Scaling factor for the squared outputs term.
        mu (float): Constant term added to the loss.
        w (float): Exponent applied to the targets.
    """
    def __init__(self, reduction: str = 'mean', lambda_: float = 1.0, mu: float = 0.25, w: float = 1.0):
        super(DistanceMatrixLoss, self).__init__()
        # Ensure the reduction method is valid
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'."
        self.reduction = reduction
        self.lambda_ = lambda_
        self.mu = mu
        self.w = w

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the Distance Matrix loss.

        Args:
            outputs (torch.Tensor): Predicted values (e.g., model outputs).
            targets (torch.Tensor): Ground truth values (distance matrices with values ranging from 0 to 1).

        Returns:
            torch.Tensor: Computed loss based on the specified reduction method.
        """
        # Compute the loss using the formula: lambda_ * (outputs^2 * targets^w + mu)
        loss = self.lambda_ * (outputs ** 2 * targets ** self.w + self.mu)
        
        # Flatten spatial dimensions (e.g., D, H, W for 3D or H, W for 2D) and sum over them
        loss = loss.view(loss.size(0), -1).sum(dim=-1)  # Sum over spatial dimensions
        
        # Apply the specified reduction method
        if self.reduction == 'mean':
            # Compute the mean loss over the batch
            return loss.mean()
        elif self.reduction == 'sum':
            # Compute the total loss by summing over the batch
            return loss.sum()
        elif self.reduction == 'none':
            # Return the per-sample loss without any reduction
            return loss
