import torch
import torch.nn as nn
import torch.nn.functional as F

def to_probability_distributions(volume):
    B, C, D, H, W = volume.shape
    # softmax over voxel domain: do reshape softmax for stability & correctness
    voxels = volume.view(B, C, -1)
    probs = F.softmax(voxels, dim=-1).view(B, C, D, H, W)
    return probs

class MeanSquaredErrorLoss(nn.Module):
    """
    Mean Squared Error (MSE) loss function.
    This class allows for flexible reduction methods: 'mean', 'sum', or 'none'.

    Supports both heatmap and coordinate training tasks.

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

class SoftmaxCrossEntropyLoss(nn.Module):
    """
    Softmax Cross-Entropy Loss function.
    The loss measures the dissimilarity between the predicted and target distributions.
    
    Supports flexible reduction methods: 'mean', 'sum', or 'none'.

    """
    def __init__(self, reduction: str = 'mean', eps: float = 1e-20):
        """
        Initializes the Softmax Cross-Entropy Loss module.

        Args:
            reduction (str): Specifies the reduction method to apply.
                - 'mean': Returns the mean of the loss.
                - 'sum': Returns the sum of the loss.
                - 'none': Returns the loss without reduction.
            eps (float): A small value to avoid numerical instability (e.g., log(0)).
        """
        super(SoftmaxCrossEntropyLoss, self).__init__()
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'."
        self.reduction = reduction
        self.eps = eps

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the Softmax Cross-Entropy Loss.

        Args:
            outputs (torch.Tensor): Predicted logits (unnormalized scores).
            targets (torch.Tensor): Ground truth probabilities (one-hot encoded or soft labels).

        Returns:
            torch.Tensor: Computed loss.
        """
        # Normalize each landmark (channel) so that its sum over spatial dims is 1
        targets = to_probability_distributions(targets)
        outputs = to_probability_distributions(outputs)

        # Compute the element-wise cross-entropy loss
        loss = - targets * torch.log(outputs + self.eps)
        # Sum the loss over spatial dimensions (e.g., D, H, W for 3D or H, W for 2D)
        loss = loss.view(loss.size(0), -1).sum(dim=-1)

        # Apply the specified reduction method
        if self.reduction == 'mean':
            # Return the mean loss over the batch
            return loss.mean()
        elif self.reduction == 'sum':
            # Return the sum of all loss values over the batch
            return loss.sum()
        elif self.reduction == 'none':
            # Return the per-sample loss without reduction
            return loss

class KullbackLeiblerDivLoss(nn.Module):
    """
    Computes element-wise KLD: targets * (log(targets) - log(outputs))

    This loss measures the Kullback-Leibler divergence between two probability distributions.
    
    """

    def __init__(self, reduction: str = 'mean', eps: float = 1e-20):
        """
        Initializes the Kullback-Leibler divergence loss.

        Args:
            reduction (str): Specifies the reduction method to apply.
                - 'mean': Returns the mean of the loss.
                - 'sum': Returns the sum of the loss.
                - 'none': Returns the loss without reduction.
            eps (float): A small value to avoid numerical instability (e.g., log(0)).
        """
        super(KullbackLeiblerDivLoss, self).__init__()
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'."
        self.reduction = reduction
        self.eps = eps

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the Kullback-Leibler divergence loss.

        Args:
            outputs (torch.Tensor): Predicted values (logits).
            targets (torch.Tensor): Ground truth probability distributions.

        Returns:
            torch.Tensor: Computed loss.
        """
        # Normalize each landmark (channel) so that its sum over spatial dims is 1
        targets = to_probability_distributions(targets)
        outputs = to_probability_distributions(outputs)

        loss = targets * (torch.log(targets) - torch.log(outputs))
        # Sum over spatial dimensions (e.g., D, H, W for 3D or H, W for 2D)
        loss = loss.view(loss.size(0), -1).sum(dim=-1)

        # Apply the specified reduction method
        if self.reduction == 'mean':
            return loss.mean()  # Return the mean loss over the batch
        elif self.reduction == 'sum':
            return loss.sum()  # Return the sum of all loss values
        elif self.reduction == 'none':
            return loss  # Return the per-sample loss without reduction

class DistanceMatrixLoss(nn.Module):
    """
    Compute the loss using the formula: lambda_ * (outputs^2 * targets^w + mu)

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

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the Distance Matrix loss.

        Args:
            outputs (torch.Tensor): Predicted values (e.g., model outputs).
            targets (torch.Tensor): Ground truth values (distance matrices with values ranging from 0 to 1).

        Returns:
            torch.Tensor: Computed loss based on the specified reduction method.
        """
        # Normalize the targets to ensure they sum to 1 (softmax-like behavior)
        dist_ms = 1 - targets
        # Normalize each landmark (channel) so that its sum over spatial dims is 1
        targets = to_probability_distributions(targets)
        outputs = to_probability_distributions(outputs)

        loss = self.lambda_ * (outputs * dist_ms ** self.w + self.mu)
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
    This loss function computes a custom loss that combines the Earth Mover's Distance (EMD) 
    regularization with a softmax cross-entropy term. It is designed to measure the similarity 
    between predicted and ground truth probability distributions.

    The loss is computed as:
        Loss = -targets * log(outputs + eps) + lambda_ * (outputs * (1 - targets)^w + mu)

    where:
        - `targets` are the ground truth probability distributions.
        - `outputs` are the predicted probability distributions.
        - `eps` is a small constant to avoid numerical instability in log calculations.
        - `lambda_` is a scaling factor for the regularization term.
        - `mu` is a constant term added to the regularization.
        - `w` is the exponent applied to the distance matrix (1 - targets).

    Args:
        reduction (str): Specifies the reduction method to apply.
            - 'mean': Returns the mean of the loss.
            - 'sum': Returns the sum of the loss.
            - 'none': Returns the loss without reduction.
        lambda_ (float): Scaling factor for the regularization term.
        mu (float): Constant term added to the regularization term.
        w (float): Exponent applied to the distance matrix (1 - targets).
    """

    def __init__(self, reduction: str = 'mean', lambda_: float = 0.5, mu: float = 0, w: float = 1.0, eps: float = 1e-20):
        super(EMDRegularizedLoss, self).__init__()
        # Ensure the reduction method is valid
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'."
        self.reduction = reduction
        self.lambda_ = lambda_
        self.mu = mu
        self.w = w
        self.eps = eps

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the Distance Matrix loss.

        Args:
            outputs (torch.Tensor): Predicted values (e.g., model outputs).
            targets (torch.Tensor): Ground truth values (distance matrices with values ranging from 0 to 1).

        Returns:
            torch.Tensor: Computed loss based on the specified reduction method.
        """

        # Normalize the targets to ensure they sum to 1 (softmax-like behavior)
        dist_ms = 1 - targets
        # Normalize each landmark (channel) so that its sum over spatial dims is 1
        targets = to_probability_distributions(targets)
        outputs = to_probability_distributions(outputs)

        # Compute softmax cross entropy: targets * (log(outputs))
        loss1 = - targets * torch.log(outputs + self.eps)
        # Compute the regularization using the formula: outputs^2 * targets^2
        loss2 = outputs * dist_ms ** 2 + self.eps

        loss = self.lambda_ * loss1 + (1 - self.lambda_) * loss2    
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

class L1_EMDRegularizedLoss(nn.Module):
    """
    This loss function computes a custom loss that combines the L1 and Earth Mover's Distance (EMD) 
    regularizations with a softmax cross-entropy term. It is designed to measure the similarity 
    between predicted and ground truth probability distributions.

    The loss is computed as:
        Loss = -targets * log(outputs + eps) + lambda_ * (outputs * (1 - targets)^w + mu)

    where:
        - `targets` are the ground truth probability distributions.
        - `outputs` are the predicted probability distributions.
        - `eps` is a small constant to avoid numerical instability in log calculations.
        - `lambda_` is a scaling factor for the regularization term.
        - `mu` is a constant term added to the regularization.
        - `w` is the exponent applied to the distance matrix (1 - targets).

    Args:
        reduction (str): Specifies the reduction method to apply.
            - 'mean': Returns the mean of the loss.
            - 'sum': Returns the sum of the loss.
            - 'none': Returns the loss without reduction.
        lambda_ (float): Scaling factor for the regularization term.
        mu (float): Constant term added to the regularization term.
        w (float): Exponent applied to the distance matrix (1 - targets).
    """

    def __init__(self, reduction: str = 'mean', lambda_: float = 1, mu: float = 0, w: float = 1.0, eps: float = 1e-20):
        super(L1_EMDRegularizedLoss, self).__init__()
        # Ensure the reduction method is valid
        assert reduction in ['mean', 'sum', 'none'], "Reduction must be 'mean', 'sum', or 'none'."
        self.reduction = reduction
        self.lambda_ = lambda_
        self.mu = mu
        self.w = w
        self.eps = eps

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the Distance Matrix loss.

        Args:
            outputs (torch.Tensor): Predicted values (e.g., model outputs).
            targets (torch.Tensor): Ground truth values (distance matrices with values ranging from 0 to 1).

        Returns:
            torch.Tensor: Computed loss based on the specified reduction method.
        """

        # Normalize the targets to ensure they sum to 1 (softmax-like behavior)
        dist_ms = 1 - targets
        # Normalize each landmark (channel) so that its sum over spatial dims is 1
        targets = to_probability_distributions(targets)
        outputs = to_probability_distributions(outputs)

        # Compute softmax cross entropy: targets * (log(outputs))
        loss1 = - targets * torch.log(outputs + self.eps)
        # Compute the regularization using the formula: lambda_ * (outputs^2 * targets^w + mu)
        loss2 = self.lambda_ * (outputs * dist_ms ** self.w + self.mu)
        # Compute the element-wise squared difference
        loss3 = (outputs - targets) ** 2
        
        loss = loss1 + loss2 + loss3
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
