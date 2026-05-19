import torch
import torch.nn as nn
import torch.nn.functional as F

def _softmax(volume):
    B, C, D, H, W = volume.shape
    # Apply softmax over voxel domain.
    voxels = volume.reshape(B, C, -1) # Reshape to voxels
    probs = F.softmax(voxels, dim=-1).view(B, C, D, H, W)
    return probs

def _log_softmax(volume):
    B, C, D, H, W = volume.shape
    # Apply softmax over voxel domain.
    voxels = volume.reshape(B, C, -1) # Reshape to voxels
    probs = F.log_softmax(voxels, dim=-1).view(B, C, D, H, W)
    return probs

class baseLoss(nn.Module):
    def __init__(self, _lambda, reduction: str = 'mean',**args):
        super(baseLoss, self).__init__()
        # Ensure the reduction method is valid
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'."
        self.reduction = reduction
        self._lambda = _lambda

    def __str__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({args})"
        
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.formula(outputs, targets)
        # {?} Average, instead of sum. Let's see what happens.
        loss = loss.view(loss.size(0), -1).sum(dim=-1)  # Sum over spatial dimensions.
        loss *= self._lambda
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


class SSELoss(baseLoss):
    def __init__(self, reduction: str = 'mean', _lambda: float = 1.0, sigmoid: bool = False, **args):
        super(SSELoss, self).__init__(
            _lambda=_lambda,
            reduction=reduction,
            **args
        )
        self.sigmoid = sigmoid

    def formula(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # {!} Given targets as in shape B, C, T, D, H, W where T is the target channel.
        # distance = targets[:, :, 1]
        targets = targets[:, :, 0]
        # Compute mse regularization.
        if self.sigmoid:
            outputs = F.sigmoid(outputs)
        loss = (outputs - targets) ** 2
        return loss

class SoftmaxCELoss(baseLoss):
    def __init__(self, reduction: str = 'mean', _lambda: float = 1.0, eps: float = 1e-15, normalize_targets: bool = True, **args) -> torch.Tensor:
        super(SoftmaxCELoss, self).__init__(
            _lambda=_lambda,
            reduction=reduction,
            **args
        )
        self.eps = eps
        self.normalize_targets = normalize_targets

    def formula(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # {!} Given targets as in shape B, C, T, D, H, W where T is the target channel.
        # distance = targets[:, :, 1]
        targets = targets[:, :, 0]
        # {!} 'targets' voxel values are in between 0-1, depicting a probability map. 
        # {!} 'outputs' voxel values are unbounded as they are raw outputs from the model. 
        # Softmax converts logits into a normalized spatial probability distribution.
        # This guarantees positive probabilities summing to 1.
        # {?} Should we only apply softmax to outputs? 
        # {!} Normalized target → distribution matching, where as unnormalized target → weighted penalty field.
        if self.normalize_targets:
            targets = _softmax(targets)
        # Compute cross entropy regularization.
        loss = - targets * _log_softmax(outputs + self.eps)
        return loss

class EucEMDLoss(baseLoss):
    def __init__(self, reduction: str = 'mean', _lambda: float = 1.0, w: int = 2, **args) -> torch.Tensor:
        super(EucEMDLoss, self).__init__(
            _lambda=_lambda,
            reduction=reduction,
            **args
        )
        self.w = w # {!} This adjusts the sensitivity of distance matrix.

    def formula(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # {!} Given targets as in shape B, C, T, D, H, W where T is the target channel.
        distance = targets[:, :, 1]
        targets = targets[:, :, 0]
        # {!} 'targets' voxel values are floats in between [0-1], depicting a probability map. 
        # {!} 'outputs' voxel values are unbounded as they are raw outputs from the model. 
        # {!} 'distance' voxel values are Euclidean distances of coordinates respect to target coordinate.
        # {!} The purpose of softmax is to ensure sum(n)_N=1 condition for EMD to hold. 
        # {?} There shouldn't be a need to normalize, but let's try.
        # targets = targets / (targets.sum(dim=(-1,-2,-3), keepdim=True) + self.eps)
        # Compute Euclidean distance-based EMD regularization.
        # {?} Should we multiply by non-zero 'targets' mask?
        loss = _softmax(outputs) * distance.pow(self.w)  
        return loss
    
class MultiNoiseLoss(nn.Module):
    """
    Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (Kendall et al; CVPR 2018).
    Ref: https://github.com/murnanedaniel/Dynamic-Loss-Weighting/blob/master/loss_models.py 
    """
    def __init__(self, n_losses: int, device: str = 'cuda'):
        super(MultiNoiseLoss, self).__init__()
        self.noise_params = nn.Parameter(torch.rand(n_losses, device=device))
    
    def forward(self, losses: list) -> torch.tensor:
        """
        Computes the total loss as a function of a list of classification losses.

        Each loss coeff is of the form: :math:`\frac{1}{\sqrt{\eta_i}} \cdot \ell_i + \log(\eta_i)`
        Total loss: :math:`\ell = \sum_{i=1}^{k} \left\[ \frac{1}{\sqrt{\eta_i}} \cdot \ell_i + \log(\eta_i) \right\]`
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            
            loss = loss.squeeze() # ensures loss is scalar (shape []); handling loss tensor shapes like [1,1]
            total_loss += (1/torch.square(self.noise_params[i]))*loss + torch.log(self.noise_params[i]) # {?} Check if the equation is correct: arxiv.org/abs/1705.07115
        
        return total_loss