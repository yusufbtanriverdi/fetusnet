import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_softmax(volume):
    B, C, D, H, W = volume.shape
    # Apply softmax over voxel domain.
    voxels = volume.view(B, C, -1) # {!} Reshape to voxels (Assumption: B=1 & C=1)
    probs = F.softmax(voxels, dim=-1).view(B, C, D, H, W)
    return probs

class MSELoss(nn.Module):
    def __init__(self, reduction: str = 'mean', sigmoid: bool = False, **args):
        super(MSELoss, self).__init__()
        # Ensure the reduction method is valid
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'."
        self.reduction = reduction
        self.sigmoid = sigmoid

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # {!} Given targets as in shape B, C, T, D, H, W where T is the target channel.
        # distance = targets[:, :, 1]
        targets = targets[:, :, 0]
        # Compute mse regularization.
        if self.sigmoid:
            outputs = F.sigmoid(outputs)
        loss = (outputs - targets) ** 2
        # {!} Average, instead of sum. Let's see what happens.
        loss = loss.view(loss.size(0), -1).mean(dim=-1)  # Average over spatial dimensions.
        # Apply the specified reduction method.
        if self.reduction == 'mean':
            # Compute the mean loss over the batch.
            return loss.mean()
        elif self.reduction == 'sum':
            # Compute the total loss by summing over the batch.
            return loss.sum()
        elif self.reduction == 'none':
            # Return the per-sample loss without any reduction.
            return loss

class SSELoss(nn.Module):
    def __init__(self, reduction: str = 'mean', sigmoid: bool = False, **args):
        super(MSELoss, self).__init__()
        # Ensure the reduction method is valid
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'."
        self.reduction = reduction
        self.sigmoid = sigmoid

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # {!} Given targets as in shape B, C, T, D, H, W where T is the auxiliary target dimension.
        # distance = targets[:, :, 1]
        targets = targets[:, :, 0]
        # Compute mse regularization.
        if self.sigmoid:
            outputs = F.sigmoid(outputs)
        loss = (outputs - targets) ** 2
        # {!} This was your usual take, previously. 
        loss = loss.view(loss.size(0), -1).sum(dim=-1)  # Sum over spatial dimensions.
        # Apply the specified reduction method.
        if self.reduction == 'mean':
            # Compute the mean loss over the batch.
            return loss.mean()
        elif self.reduction == 'sum':
            # Compute the total loss by summing over the batch.
            return loss.sum()
        elif self.reduction == 'none':
            # Return the per-sample loss without any reduction.
            return loss


class SoftmaxCELoss(nn.Module):
    def __init__(self, reduction: str = 'mean', eps: float = 1e-30, normalize_targets: bool = True, **args) -> torch.Tensor:
        super(SoftmaxCELoss, self).__init__()
        # Ensure the reduction method is valid
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'."
        self.reduction = reduction
        self.eps = eps
        self.normalize_targets = normalize_targets

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # {!} Given targets as in shape B, C, T, D, H, W where T is the target channel.
        # distance = targets[:, :, 1]
        targets = targets[:, :, 0]
        # {!} 'targets' voxel values are in between 0-1, depicting a probability map. 
        # {!} 'outputs' voxel values are unbounded as they are raw outputs from the model. 
        # {!} The purpose of softmax is to ensure conditions (mathematical constraint of log10, penalization, etc.) in the 2nd term.
        # {?} Should we only apply softmax to outputs? 
        # {!} Normalized target → distribution matching, where as unnormalized target → weighted penalty field.
        if self.normalize_targets:
            targets = targets / (targets.sum(dim=(-1,-2,-3), keepdim=True) + self.eps)
        outputs = apply_softmax(outputs)
        # Compute cross entropy regularization.
        loss = - targets * torch.log(outputs + self.eps)
        loss = loss.view(loss.size(0), -1).sum(dim=-1)  # Sum over spatial dimensions.
        # Apply the specified reduction method.
        if self.reduction == 'mean':
            # Compute the mean loss over the batch.
            return loss.mean()
        elif self.reduction == 'sum':
            # Compute the total loss by summing over the batch.
            return loss.sum()
        elif self.reduction == 'none':
            # Return the per-sample loss without any reduction.
            return loss

class EucEMDLoss(nn.Module):
    def __init__(self, reduction: str = 'mean', w: int = 2, **args) -> torch.Tensor:
        super(EucEMDLoss, self).__init__()
        # Ensure the reduction method is valid
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'."
        self.reduction = reduction
        self.w = w # {!} This adjusts the sensitivity of distance matrix.

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # {!} Given targets as in shape B, C, T, D, H, W where T is the target channel.
        distance = targets[:, :, 1]
        targets = targets[:, :, 0]
        # {!} 'targets' voxel values are floats in between [0-1], depicting a probability map. 
        # {!} 'outputs' voxel values are unbounded as they are raw outputs from the model. 
        # {!} 'distance' voxel values are Euclidean distances of coordinates respect to target coordinate.
        # {!} The purpose of softmax is to ensure sum(n)_N=1 condition for EMD to hold. 
        outputs = apply_softmax(outputs)
        # Compute Euclidean distance-based EMD regularization.
        loss = outputs * (targets > 0) * distance ** self.w  # {!} Multiply by non-zero 'targets' mask.
        loss = loss.view(loss.size(0), -1).sum(dim=-1)  # Sum over spatial dimensions.
        # Apply the specified reduction method.
        if self.reduction == 'mean':
            # Compute the mean loss over the batch.
            return loss.mean()
        elif self.reduction == 'sum':
            # Compute the total loss by summing over the batch.
            return loss.sum()
        elif self.reduction == 'none':
            # Return the per-sample loss without any reduction.
            return loss