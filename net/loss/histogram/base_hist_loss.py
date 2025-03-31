import torch
from torch import nn, Tensor
from abc import ABC, abstractmethod
from utils import norm_min_max_distributions, discrete_intensity_histogram, triangular_histogram_with_linear_slope


class BaseHistLoss(nn.Module, ABC):
    """
    Base class for all Loss with histograms

    Args:
        bins (int, optional): .Default: `128`
        alpha (float, optional): parameter for regularization. Default: `0`

    Shape:
        - pos_input: set of positive points, (N, *)
        - neg_input: set of negative points, (M, *)
        - output: scalar
    """
    def __init__(self, bins: int = 128, alpha: float = 0):
        super(BaseHistLoss, self).__init__()
        self.bins = bins
        self._max_val = 1
        self._min_val = 0
        self.alpha = alpha
        self.delta = (self._max_val - self._min_val) / (bins - 1)
        self.t = torch.arange(self._min_val, self._max_val + self.delta, step=self.delta)

    def compute_histogram(self, inputs: Tensor) -> Tensor:
        # return triangular_histogram_with_linear_slope(inputs, self.t, self.delta)
        return discrete_intensity_histogram(inputs, self.bins)
    
    @abstractmethod
    def forward(self, outputs: Tensor, targets: Tensor):
        # outputs, targets = norm_min_max_distributions(outputs), norm_min_max_distributions(targets)
        pass