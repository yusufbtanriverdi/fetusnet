import torch
from torch import Tensor

def pdf(*distributions: Tensor):
    return (d / d.sum() for d in distributions)

def cdf(*distributions: Tensor):
    return (d / d.cumsum(dim=-1) for d in distributions)


def discrete_intensity_histogram(inputs: torch.Tensor, bins: int):
    """
    Computes a discrete histogram of intensity values efficiently.

    Args:
        inputs (Tensor): Input tensor (e.g., an image or heatmap)
        bins (int): Number of bins (e.g., for pixel intensity range 0-255, use 256)

    Returns:
        Tensor: Histogram of intensity values
    """
    inputs = inputs.view(-1)  # Flatten the tensor
    hist = torch.histc(inputs, bins=bins, min=0, max=1)
    return hist

def triangular_histogram_with_linear_slope(inputs: Tensor, t: Tensor, delta: float):
    """
    Function that calculates a histogram from an article
    [Learning Deep Embeddings with Histogram Loss](https://arxiv.org/pdf/1611.00822.pdf)
    Args:
        input (Tensor): tensor that contains the data
        t (Tensor): tensor that contains the nodes of the histogram
        delta (float): step in histogram
    """
    inputs = inputs.view(-1)
    t = t.cuda()

    # first condition of the second equation of the paper
    x = inputs.unsqueeze(0) - t.unsqueeze(1) + delta
    m = torch.zeros_like(x)
    m[(0 <= x) & (x <= delta)] = 1
    a = torch.sum(x * m, dim=1) / (delta * len(inputs))

    # second condition of the second equation of the paper
    x = t.unsqueeze(0) - inputs.unsqueeze(1) + delta
    m = torch.zeros_like(x)
    m[(0 <= x) & (x <= delta)] = 1
    b = torch.sum(x * m, dim=0) / (delta * len(inputs))

    return torch.add(a, b)

# TO BE TESTED
def hard_joint_histogram(x: torch.Tensor, y: torch.Tensor, bins: int):
    x, y = x.view(-1), y.view(-1)  # Flatten tensors
    # Compute bin edges
    bin_edges = torch.linspace(0, 1, bins + 1, device=x.device)
    # Initialize histogram
    m = torch.zeros(bins, bins, device=x.device)

    # Create masks for x and y values
    for i in range(bins):
        x_mask = (x >= bin_edges[i]) & (x < bin_edges[i + 1])  # Mask for x-bin
        for j in range(bins):
            y_mask = (y >= bin_edges[j]) & (y < bin_edges[j + 1])  # Mask for y-bin
            # Count values in both x and y masks
            m[i, j] = torch.sum(x_mask & y_mask)  # Element-wise AND
    return m

def soft_joint_histogram(x: torch.Tensor, y: torch.Tensor, bins: int = 10, sigma: float = 0.01):
    """
    Differentiable 2D joint histogram using Gaussian soft binning.

    Args:
        x (Tensor): First input tensor.
        y (Tensor): Second input tensor.
        bins (int): Number of bins.
        sigma (float): Bandwidth for Gaussian kernel.

    Returns:
        Tensor: Soft joint histogram (bins x bins)
    """
    x, y = x.view(-1), y.view(-1)  # Flatten
    assert x.shape == y.shape

    device = x.device
    bin_centers = torch.linspace(0, 1, bins, device=device, dtype=x.dtype)
    
    # Expand for broadcasting
    x = x.unsqueeze(1)  # [N, 1]
    y = y.unsqueeze(1)  # [N, 1]
    cx = bin_centers.unsqueeze(0)  # [1, B]
    cy = bin_centers.unsqueeze(0)  # [1, B]

    # Compute soft assignments using Gaussian kernel
    wx = torch.exp(-0.5 * ((x - cx) / sigma) ** 2)  # [N, B]
    wy = torch.exp(-0.5 * ((y - cy) / sigma) ** 2)  # [N, B]

    # Normalize each row (optional for probability-like behavior)
    # wx = wx / (wx.sum(dim=1, keepdim=True) + 1e-8)
    # wy = wy / (wy.sum(dim=1, keepdim=True) + 1e-8)

    # Outer product for each pair, then sum
    hist2d = torch.einsum('nb,nc->bc', wx.to(torch.float64), wy)  # [B, B]

    return hist2d  # Normalize to make it a PDF
