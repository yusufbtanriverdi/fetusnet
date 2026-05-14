import torch
import torch.nn as nn
import torch.nn.functional as F
from net.plot.heatmaps import plot_heatmaps_slices_from_coord

def to_probability_distributions(volume):
    B, C, D, H, W = volume.shape
    # softmax over voxel domain: do reshape softmax for stability & correctness
    voxels = volume.view(B, C, -1)
    probs = F.softmax(voxels, dim=-1).view(B, C, D, H, W)
    return probs


class EucEMDRegularizedLoss(nn.Module):
    """
    This loss function computes a custom loss that combines the Earth Mover's Distance (EMD) 
    regularization with a softmax cross-entropy term. It is designed to measure the similarity 
    between predicted and ground truth probability distributions.

    The loss is computed as:
        Loss = -targets * log(outputs + eps) + alpha_ * (outputs * (1 - targets)^w + mu)

    where:
        - `targets` are the ground truth probability distributions.
        - `outputs` are the predicted probability distributions.
        - `eps` is a small constant to avoid numerical instability in log calculations.
        - `alpha_` is a scaling factor for the regularization term.
        - `mu` is a constant term added to the regularization.
        - `w` is the exponent applied to the distance matrix (1 - targets).

    Args:
        reduction (str): Specifies the reduction method to apply.
            - 'mean': Returns the mean of the loss.
            - 'sum': Returns the sum of the loss.
            - 'none': Returns the loss without reduction.
        alpha_ (float): Scaling factor for the regularization term.
        mu (float): Constant term added to the regularization term.
        w (float): Exponent applied to the distance matrix (1 - targets).
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-20, 
                 a: float = 1.0, 
                 b: float = 1.0, 
                 c: float = 1.0):
        super(EucEMDRegularizedLoss, self).__init__()
        # Ensure the reduction method is valid
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'."
        self.reduction = reduction
        self.a, self.b, self.c = a, b, c
        self.eps = eps

    def __str__(self):
        return f"EucEMDRegularizedLoss( \n reduction: {self.reduction} \n eps: {self.eps} \n  a: {self.a} \n  b: {self.b} \n c: {self.c} \n)"

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the Distance Matrix loss.

        Args:
            outputs (torch.Tensor): Predicted values (e.g., model outputs).
            targets (torch.Tensor): Ground truth values (distance matrices with values ranging from 0 to 1).

        Returns:
            torch.Tensor: Computed loss based on the specified reduction method.
        """
        # Given targets as in shape B, C, T, D, H, W where T is the target channel.
        distance = targets[:, :, 1]
        targets = targets[:, :, 0]
        # Compute mse regularization
        # outputs_ = F.sigmoid(outputs)
        mse = (outputs - targets) ** 2
        # Normalize each landmark (channel) so that its sum over spatial dims is 1
        targets = to_probability_distributions(targets)
        outputs = to_probability_distributions(outputs)
        # Compute softmax cross entropy: targets * (log(outputs))
        ce = - targets * torch.log(outputs + self.eps)
        # Compute the EMD regularization from mitral paper
        # loss2 = outputs * (distance ** 2) #formula v1.
        emd = outputs * (targets > 0) * distance**2 #formula v2.

        loss = self.a * ce + self.b * emd + self.c * mse    
        # coords = torch.nonzero(targets[0][1] == targets[0][1].max(), as_tuple=False).float().detach().cpu().numpy()[0]
        # plot_heatmaps_slices_from_coord([loss[0][1], loss1[0][1], loss2[0][1], loss3[0][1], targets[0][1], outputs[0][1], distance[0][1]], coords, titles=['Overall Loss', 'Softmax CE', 'EMD-Regularization', 'Mean Squared Error', 'Target', 'Predicted', 'Distance Matrix'])

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