import torch
import torch.nn as nn
from net.loss.utils import plot_histograms_and_stats, imshow_target_distance_matrices

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
    def __init__(self, reduction: str = 'mean', lambda_: float = 1, mu: float = 0, w: float = 1.0):
        super(DistanceMatrixLoss, self).__init__()
        # Ensure the reduction method is valid
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'."
        self.reduction = reduction
        self.lambda_ = lambda_
        self.mu = mu
        self.w = w

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, flag_visualize=False) -> torch.Tensor:
        """
        Forward pass to compute the Distance Matrix loss.

        Args:
            outputs (torch.Tensor): Predicted values (e.g., model outputs).
            targets (torch.Tensor): Ground truth values (distance matrices with values ranging from 0 to 1).

        Returns:
            torch.Tensor: Computed loss based on the specified reduction method.
        """
        # Normalize the targets to ensure they sum to 1 (softmax-like behavior)
        # targets = targets / (targets.view(targets.size(0), -1).sum(dim=-1, keepdim=True))
        dist_ms = 1 - targets
        targets = targets / (targets.view(targets.size(0), -1).sum(dim=-1, keepdim=True))
        # dist_ms = dist_ms / (dist_ms.view(dist_ms.size(0), -1).sum(dim=-1, keepdim=True))

        # Apply softmax to the flattened outputs
        outputs = torch.nn.functional.softmax(outputs.view(outputs.size(0), -1), dim=-1).view_as(outputs)

        # Call the function to visualize
        # plot_histograms_and_stats(outputs, targets)

        # Compute the loss using the formula: lambda_ * (outputs^2 * targets^w + mu)
        loss = self.lambda_ * (outputs * dist_ms ** self.w + self.mu)

        # Call the function to visualize the target distance matrix
        if flag_visualize:
            imshow_target_distance_matrices(outputs, targets, dist_ms, loss, titles=['Probability Maps', 'GT Heatmap', 'Distance Matrix', 'Loss'])

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

class EMDRegularizedLoss(nn.Module):
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
    def __init__(self, reduction: str = 'mean', lambda_: float = 1, mu: float = 0, w: float = 1.0, eps: float = 1e-20):
        super(EMDRegularizedLoss, self).__init__()
        # Ensure the reduction method is valid
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'."
        self.reduction = reduction
        self.lambda_ = lambda_
        self.mu = mu
        self.w = w
        self.eps = eps

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, flag_visualize=False) -> torch.Tensor:
        """
        Forward pass to compute the Distance Matrix loss.

        Args:
            outputs (torch.Tensor): Predicted values (e.g., model outputs).
            targets (torch.Tensor): Ground truth values (distance matrices with values ranging from 0 to 1).

        Returns:
            torch.Tensor: Computed loss based on the specified reduction method.
        """
        # Normalize the targets to ensure they sum to 1 (softmax-like behavior)
        # targets = targets / (targets.view(targets.size(0), -1).sum(dim=-1, keepdim=True))
        dist_ms = 1 - targets
        targets = targets / (targets.view(targets.size(0), -1).sum(dim=-1, keepdim=True))
        # dist_ms = dist_ms / (dist_ms.view(dist_ms.size(0), -1).sum(dim=-1, keepdim=True))

        # Apply softmax to the flattened outputs
        outputs = torch.nn.functional.softmax(outputs.view(outputs.size(0), -1), dim=-1).view_as(outputs)

        loss1 = - targets * torch.log(outputs + self.eps)
        
        # Call the function to visualize
        # plot_histograms_and_stats(outputs, targets)

        # Compute the loss using the formula: lambda_ * (outputs^2 * targets^w + mu)
        loss2 = self.lambda_ * (outputs * dist_ms ** self.w + self.mu)

        # Call the function to visualize the target distance matrix
        if flag_visualize:
            imshow_target_distance_matrices(outputs, targets, dist_ms, loss1, loss2, titles=['Probability Maps', 'GT Heatmap', 'Distance Matrix', 'Softmax CE Loss', 'EMD Loss'])

        # Flatten spatial dimensions (e.g., D, H, W for 3D or H, W for 2D) and sum over them
        loss2 = loss2.view(loss2.size(0), -1).sum(dim=-1)  # Sum over spatial dimensions
        loss1 = loss1.view(loss1.size(0), -1).sum(dim=-1)  # Sum over spatial dimensions

        loss = loss1 + loss2    

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
