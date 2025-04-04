from torch import nn, Tensor
from abc import ABC, abstractmethod
from net.loss.utils import *
import torch.nn.functional as F

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
    def __init__(self, bins: int = 128, sigma: float = 0.01):
        super(BaseHistLoss, self).__init__()
        self.bins = bins
        self._max_val = 1
        self._min_val = 0
        self.sigma = sigma
        
    def compute_joint_histogram(self, x: Tensor, y: Tensor):
        return soft_joint_histogram(x, y, self.bins, self.sigma)

    @abstractmethod
    def forward(self, outputs: Tensor, targets: Tensor):
        outputs = F.sigmoid(outputs)
        pass

