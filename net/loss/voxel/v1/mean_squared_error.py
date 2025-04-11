import torch
import torch.nn as nn

class MeanSquaredErrorLoss(nn.Module):
    """
    Custom implementation of the Mean Squared Error (MSE) loss function.
    This class allows for flexible reduction methods: 'mean', 'sum', or 'none'.
    """
    def __init__(self, reduction: str = 'mean'):
        """
        Initializes the MeanSquaredErrorLoss module.

        Args:
            reduction (str): Specifies the reduction method to apply to the loss.
                - 'mean': Returns the mean of the loss values.
                - 'sum': Returns the sum of the loss values.
                - 'none': Returns the loss values without any reduction.
        """
        super(MeanSquaredErrorLoss, self).__init__()
        
        # Ensure the reduction method is valid
        assert reduction in ['mean', 'sum', 'none'], \
            "Reduction must be one of 'mean', 'sum', or 'none'."
        
        self.reduction = reduction

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the Mean Squared Error (MSE) loss.

        Args:
            outputs (torch.Tensor): Predicted values (model outputs).
            targets (torch.Tensor): Ground truth values (labels).

        Returns:
            torch.Tensor: The computed loss, reduced based on the specified method.
        """
        # Compute the element-wise squared difference
        loss = (outputs - targets) ** 2
        
        # Flatten spatial dimensions (e.g., D, H, W for 3D or H, W for 2D) and sum over them
        # This results in a per-sample loss
        loss = loss.view(loss.size(0), -1).sum(dim=-1)
        
        # Apply the specified reduction method
        if self.reduction == 'mean':
            # Compute the mean loss across the batch
            return loss.mean()
        elif self.reduction == 'sum':
            # Compute the total loss across the batch
            return loss.sum()
        elif self.reduction == 'none':
            # Return the per-sample loss without reduction
            return loss
