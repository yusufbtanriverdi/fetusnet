import torch
from net.loss.histogram.base_hist_loss import BaseHistLoss
from torch.autograd import Variable

class TVDLoss(BaseHistLoss):
    def __init__(self, reduction: str = 'mean'):
        """
        Custom Total Variation Distance (TVD) loss as explained in the video following.

        https://www.youtube.com/watch?v=Bk84wAkunpo.

        Args:
            reduction (str): Specifies the reduction method to apply.
                - 'mean': Returns the mean of the loss.
                - 'sum': Returns the sum of the loss.
                - 'none': Returns the loss without reduction.
        """
        super(TVDLoss, self).__init__()
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
        # Assuming batch size = 1

        out_hist = self.compute_histogram(outputs).cuda()  # h_pos
        tar_hist = self.compute_histogram(targets).cuda()  # h_neg

        # import matplotlib.pyplot as plt
        # plt.plot(torch.linspace(0, 1, 128), out_hist.detach().cpu())
        # plt.plot(torch.linspace(0, 1, 128), tar_hist.detach().cpu())
        # plt.show()

        # Assume that both outputs and targets are probabilistic distributions
        # TV(P, Q) = 1/2 * norm(P-Q)
        loss = 1/2 * torch.linalg.norm(out_hist - tar_hist)

        if self.reduction == 'mean':
            # Reduce over the batch dimension and return the mean
            return loss.mean()

        elif self.reduction == 'sum':
            # Return the sum of all the loss values over the batch
            return loss.sum()

        elif self.reduction == 'none':
            # Return the per-sample loss without reduction
            return loss

