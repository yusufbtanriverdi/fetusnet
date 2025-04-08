import torch
import torch.nn
from net.loss.utils import pdf, cdf, soft_joint_histogram, target_joint_histogram

class JointHistogramLoss(torch.nn.Module):
    def __init__(self, reduction: str = 'mean', bins: int = 128, sigma=0.001):
        """
        Custom loss function that computes the joint histogram loss between predicted
        and target distributions using a soft joint histogram approach.

        Args:
            reduction (str): Specifies the reduction method to apply to the loss.
                - 'mean': Returns the mean of the loss across the batch.
                - 'sum': Returns the sum of the loss across the batch.
                - 'none': Returns the loss for each sample in the batch without reduction.
            bins (int): The number of bins to use for the joint histogram computation.
            sigma (float): Bandwidth (standard deviation) for the Gaussian kernel used in the soft histogram computation.
        """
        super(JointHistogramLoss, self).__init__()
        
        assert reduction in ['mean', 'sum', 'none'], \
            "Reduction must be 'mean', 'sum', or 'none'."
        
        self.reduction = reduction
        self.n_bins = bins
        self.sigma = sigma

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the joint histogram loss between outputs and targets.

        Args:
            outputs (torch.Tensor): The predicted values (probabilistic distributions).
            targets (torch.Tensor): The ground truth values (probabilistic distributions).
            Hyy (torch.Tensor): The diagonal of the joint histogram for the targets. This is constant for each sample.

        Returns:
            torch.Tensor: The computed loss value. The reduction method specified in the constructor is applied.
        """
        # Assuming batch size = 1 (can be adjusted for batch processing)
        # print(targets.min(), targets.max(), outputs.min(), outputs.max())
        targets = torch.clamp(targets, min=1e-20)  # Avoid log(<=0)
        outputs = torch.nn.functional.sigmoid(outputs) # Avoid log(<=0) or log(>1)
        # print(targets.min(), targets.max(), outputs.min(), outputs.max())

        Hyy = target_joint_histogram(targets, bins=self.n_bins)[1:, 1:] # constant for each sample.
        # # Convert outputs and targets to their probabilistic forms
        # outputs, targets = pdf(outputs, targets)

        # Compute the joint histogram Hxy between outputs and targets
        Hxy = soft_joint_histogram(outputs, targets, bins=self.n_bins, sigma=self.sigma)[1:, 1:]

        # print(Hxy.shape, Hyy.shape)
        # import matplotlib.pyplot as plt
        # plt.subplot(121)
        # plt.imshow(Hyy.detach().cpu())
        # plt.subplot(122)
        # plt.imshow(Hxy.detach().cpu())       
        # plt.show()
        
        # Compute M = Hyy^-1 * Hxy - I
        M = torch.inverse(Hyy) @ Hxy - torch.eye(self.n_bins-1, device=outputs.device)

        # Compute the Frobenius norm squared of M
        loss = torch.norm(M, 'fro')  / M.numel()

        # # Apply reduction across the batch dimension
        # if self.reduction == 'mean':
        #     return loss.mean()  # Return the mean loss across the batch

        # elif self.reduction == 'sum':
        #     return loss.sum()  # Return the sum of the loss across the batch

        # elif self.reduction == 'none':
        #     return loss  # Return the individual loss for each sample in the batch
        return loss