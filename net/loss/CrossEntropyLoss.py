import torch
import torch.nn as nn
from net.loss.utility import histogram_1d, histogram_2d

class DCELoss(nn.Module):
    def __init__(self, reduction: str = 'mean', eps: int =1e-20):
        """
        Custom Kullback-Leibler (DCE) loss.

        Args:
            reduction (str): Specifies the reduction method to apply.
                - 'mean': Returns the mean of the loss.
                - 'sum': Returns the sum of the loss.
                - 'none': Returns the loss without reduction.
        """
        super(DCELoss, self).__init__()
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'."
        self.reduction = reduction
        self.eps = eps

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the DCE loss.

        Args:
            outputs (torch.Tensor): Predicted values.
            targets (torch.Tensor): Ground truth values.

        Returns:
            torch.Tensor: Computed loss.
        """

        targets = torch.clamp(targets, min=self.eps)  # Avoid log(<=0)
        outputs = torch.nn.functional.sigmoid(outputs) # Avoid log(<=0) or log(>1)
        
        loss = targets * torch.log(outputs)  # Element-wise DCE
        
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


class HistDCELoss(nn.Module):
    def __init__(self, n_bins, reduction: str = 'mean'):
        """
        Custom Mean Squared Error (DCE) loss that considers histogram bins of images.

        Args:
            reduction (str): Specifies the reduction method to apply.
                - 'mean': Returns the mean of the loss.
                - 'sum': Returns the sum of the loss.
                - 'none': Returns the loss without reduction.
        """
        super(HistDCELoss, self).__init__()
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'."
        self.reduction = reduction
        self.n_bins = n_bins

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the DCE loss between 1D histograms.

        Args:
            outputs (torch.Tensor): Predicted values.
            targets (torch.Tensor): Ground truth values.

        Returns:
            torch.Tensor: Computed loss.
        """

        out_hist = histogram_1d.create_histogram(outputs, self.n_bins) # should return N x number_of_bins
        tar_hist = histogram_1d.create_histogram(targets, self.n_bins)

        outputs = torch.nn.functional.softmax(out_hist, dim=0) # Avoid log(<=0) or log(>1)
        targets = torch.nn.functional.softmax(tar_hist, dim=0) # Avoid log(<=0) or log(>1)

        loss = targets * torch.log(outputs)  # Histogram-based DCE

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

class JointDCELoss(nn.Module):
    def __init__(self, n_bins, reduction: str = 'mean'):
        """
        Custom Kullback-Leibler (DCE) loss between HIG and HGG joint histograms. 

        Args:
            reduction (str): Specifies the reduction method to apply.
                - 'mean': Returns the mean of the loss.
                - 'sum': Returns the sum of the loss.
                - 'none': Returns the loss without reduction.
        """
        super(JointDCELoss, self).__init__()
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'."
        self.reduction = reduction
        self.n_bins = n_bins

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the DCE loss.

        Args:
            outputs (torch.Tensor): Predicted values.
            targets (torch.Tensor): Ground truth values.

        Returns:
            torch.Tensor: Computed loss.
        """

        hig = histogram_2d.create_joint_histogram_fast(outputs, targets, self.n_bins) # Size N x n_bins x n_bins
        hgg = histogram_2d.create_joint_histogram_fast(outputs, outputs, self.n_bins)

        outputs = torch.nn.functional.softmax(hig, dim=0) # Avoid log(<=0) or log(>1)
        targets = torch.nn.functional.softmax(hgg, dim=0) # Avoid log(<=0) or log(>1)

        loss = targets * torch.log(outputs)  # Joint Histogram-based DCE
        
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
