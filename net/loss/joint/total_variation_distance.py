import torch
from net.loss.joint.base import BaseHistLoss
from net.loss.utils import pdf

class TVDLoss(BaseHistLoss):
    def __init__(self, reduction: str = 'mean', bins: int = 128):
        """
        Custom Total Variation Distance (TVD) loss as explained in the video following.

        https://www.youtube.com/watch?v=Bk84wAkunpo.

        Initutively, it is equivalent to MSE. However, I want to follow same notation for distribution-based losses.

        Args:
            reduction (str): Specifies the reduction method to apply.
                - 'mean': Returns the mean of the loss.
                - 'sum': Returns the sum of the loss.
                - 'none': Returns the loss without reduction.
        """
        super(TVDLoss, self).__init__()
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'."
        self.reduction = reduction
        self.bins = bins

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the MSE loss.

        Args:
            outputs (torch.Tensor): Predicted values.
            targets (torch.Tensor): Ground truth values.

        Returns:
            torch.Tensor: Computed loss.
        """
        # ¡ Assuming batch size = 1
        # ¡ Assume that both outputs and targets are probabilistic distributions
        hig = self.compute_joint_histogram(outputs, targets)
        hgg = self.compute_joint_histogram(targets, targets)
        hig, hgg = pdf(hig, hgg)

        # ¿ TV(P, Q) = 1/2 * norm(P-Q)
        loss = 1/2 * torch.linalg.norm(hig - hgg)
        
        if self.reduction == 'mean':
            # Reduce over the batch dimension and return the mean
            return loss.mean()

        elif self.reduction == 'sum':
            # Return the sum of all the loss values over the batch
            return loss.sum()

        elif self.reduction == 'none':
            # Return the per-sample loss without reduction
            return loss

