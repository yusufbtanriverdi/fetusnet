import torch
import torch.nn.functional as F

def to_probability_distributions(volume):
    D, H, W = volume.shape
    # softmax over voxel domain: do reshape softmax for stability & correctness
    voxels = volume.view(-1)
    probs = F.softmax(voxels, dim=-1).view(D, H, W)
    return probs

# def normalized_mse(p, q):
#     """
#     Compute the Normalized Mean Squared Error (NMSE) between two probability
#     distributions. Guaranteed to be in [0,1].

#     Args:
#         p (torch.Tensor): Ground truth distribution.
#         q (torch.Tensor): Predicted distribution.

#     Returns:
#         torch.Tensor: NMSE in [0,1].
#     """
#     p = to_probability_distributions(p)
#     q = to_probability_distributions(q)
#     p = p.flatten()
#     q = q.flatten()

#     mse = torch.mean((p - q) ** 2)
#     mse_max = 2.0 / p.numel()  # worst-case MSE between two one-hot distributions
#     return mse / mse_max

def kullback_leibler_divergence(p, q, eps=1e-20):
    """
    Computes the Kullback-Leibler divergence D_KL(P || Q) between two heatmaps.
    Both p and q should be probability distributions (sum to 1).
    Range: [0, ∞)
    Interpretation:
    - D_KL = 0: Perfect match between distributions.
    - D_KL > 0: The larger the value, the more dissimilar the distributions are.
    - D_KL = ∞: Distributions do not overlap at all.

    Note: KL divergence is not symmetric, i.e., D_KL(P || Q) != D_KL(Q || P).

    Args:
        p (torch.Tensor): Ground truth heatmap.
        q (torch.Tensor): Predicted heatmap.
        eps (float): Small value to avoid log(0).               
    Returns:
        torch.Tensor: KL divergence score (lower is better).
    
    """
    p = to_probability_distributions(p)
    q = to_probability_distributions(q)
    p = p.flatten()
    q = q.flatten()
    kl = torch.sum(p * torch.log((p + eps) / (q + eps)))
    return kl

# def spatial_cross_correlation(p, q, eps=1e-20):
#     """
#     Computes the spatial cross-correlation between two heatmaps.
#     Both p and q should be torch tensors of the same shape.
#     Range: [-1, 1]
#     Interpretation:
#     - corr = 1: Perfect positive correlation.
#     - corr = 0: No correlation.
#     - corr = -1: Perfect negative correlation.

#     Args:
#         p (torch.Tensor): Ground truth heatmap.
#         q (torch.Tensor): Predicted heatmap.
#         eps (float): Small value to avoid division by zero.

#     Returns:
#         torch.Tensor: Cross-correlation score.
#     """
#     p = to_probability_distributions(p)
#     q = to_probability_distributions(q)
#     p = p.flatten()
#     q = q.flatten()
#     p = (p - torch.mean(p)) / (torch.std(p) + eps)
#     q = (q - torch.mean(q)) / (torch.std(q) + eps)
#     corr = torch.sum(p * q) / (p.numel() - 1)
#     return corr

# def histogram_matching_measure(p, q, bins=10000, eps=1e-20):
#     """
#     Computes a histogram matching measure (e.g., L1 distance between normalized histograms) between two heatmaps.
#     Both p and q should be torch tensors of the same shape.
#     Range: [0, 2] (for L1 distance between histograms)
#     Interpretation:
#     - Score = 0: Perfect match between histograms.
#     - Score > 0: The larger the value, the more dissimilar the histograms are.
#     - Score = 2: Completely disjoint histograms.
    
#     Args:
#         p (torch.Tensor): Ground truth heatmap.
#         q (torch.Tensor): Predicted heatmap.
#         bins (int): Number of histogram bins.
#         eps (float): Small value to avoid division by zero.

#     Returns:
#         torch.Tensor: Histogram matching score (lower is better).
#     """

#     p = to_probability_distributions(p)
#     q = to_probability_distributions(q)
#     p_hist = torch.histc(p.flatten(), bins=bins, min=0.0, max=1.0)
#     q_hist = torch.histc(q.flatten(), bins=bins, min=0.0, max=1.0)
#     p_hist = p_hist / (torch.sum(p_hist) + eps)
#     q_hist = q_hist / (torch.sum(q_hist) + eps)
#     score = torch.sum(torch.abs(p_hist - q_hist))
#     return score
