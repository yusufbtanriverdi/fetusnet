import torch

def d_mean_mm(outputs, targets, spacings):
    """
    Computes the mean Euclidean distance (d_mean) between predicted and ground truth landmarks in mm.
    
    Assumption: Only one landmark is present in the heatmap.

    Args:
    - outputs (torch.Tensor): Predicted landmark heatmap, shape (D, H, W).
    - targets (torch.Tensor): Ground truth landmark heatmap, shape (D, H, W).
    - spacings (float): Voxel size in mm. Assuming ISO spacing for all dimensions.

    Returns:
    - torch.Tensor: Mean distance in mm over batch.
    """

    return (torch.norm(outputs - targets) * spacings).item() # to change back to mm, we multiply by spacings
