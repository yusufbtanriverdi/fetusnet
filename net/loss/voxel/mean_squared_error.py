import torch
import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        """
        Custom Mean Squared Error (MSE) loss.

        Args:
            reduction (str): Specifies the reduction method to apply.
                - 'mean': Returns the mean of the loss.
                - 'sum': Returns the sum of the loss.
                - 'none': Returns the loss without reduction.
        """
        super(MSELoss, self).__init__()
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'."
        self.reduction = reduction

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the MSE loss.

        Args:
            outputs (torch.Tensor): Predicted values.
            targets (torch.Tensor): Ground truth values.

        Returns:
            torch.Tensor: Computed loss.
        """
        loss = (outputs - targets) ** 2  # Element-wise squared difference
        
        # Sum over spatial dimensions (D, H, W for 3D, or H, W for 2D)
        loss = loss.view(loss.size(0), -1).sum(dim=-1)  # Sum over D,H,W
        
        if self.reduction == 'mean':
            # Reduce over the batch dimension and return the mean
            return loss.mean()
        
        elif self.reduction == 'sum':
            # Return the sum of all the loss values over the batch
            return loss.sum()
        
        elif self.reduction == 'none':
            # Return the per-sample loss without reduction
            return loss

