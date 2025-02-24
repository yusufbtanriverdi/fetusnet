import torch

def d_mean_mm(outputs, targets, spacings, method='argmax'):
    """
    Computes the mean Euclidean distance (d_mean) between predicted and ground truth landmarks in mm.
    
    Assumption: Only one landmark is present in the heatmap.

    Args:
    - outputs (torch.Tensor): Predicted landmark heatmap, shape (N, D, H, W).
    - targets (torch.Tensor): Ground truth landmark heatmap, shape (N, D, H, W).
    - spacings (list or torch.Tensor): Voxel size in mm [spacing_D, spacing_H, spacing_W].
    - method (str): Peak detection method. 
        - 'com' (center of mass) (recommended for better accuracy).
        - 'argmax' (hard max location).

    Returns:
    - torch.Tensor: Mean distance in mm over batch.
    """

    def get_peak_location(heatmap, method):
        """Extracts peak location from a heatmap."""
        if method == 'argmax':
            peak = torch.nonzero(heatmap == heatmap.max(), as_tuple=False).float()
            return peak[0] if peak.numel() > 0 else None

        # elif method == 'com':
        #     # Compute soft center-of-mass for smoother localization
        #     coords = torch.meshgrid(torch.arange(heatmap.shape[0]),
        #                             torch.arange(heatmap.shape[1]),
        #                             torch.arange(heatmap.shape[2]), indexing='ij')
        #     coords = torch.stack(coords, dim=-1).float().to(heatmap.device)  # Shape (D, H, W, 3)

        #     heatmap = heatmap / heatmap.sum()  # Normalize to make it a probability map
        #     com = (heatmap.unsqueeze(-1) * coords).sum(dim=(0, 1, 2))  # Weighted sum
        #     return com

    all_distances = []
    N, D, H, W = outputs.shape  # One landmark only

    for i in range(N):  # Iterate over batch
        pred_peak = get_peak_location(outputs[i], method)
        gt_peak = get_peak_location(targets[i], method)

        if pred_peak is not None and gt_peak is not None:
            pred_peak *= torch.tensor(spacings[i][0], device=pred_peak.device)
            gt_peak *= torch.tensor(spacings[i][0], device=gt_peak.device)

            distance = torch.norm(pred_peak - gt_peak)
            all_distances.append(distance)

    return torch.tensor(all_distances).mean()
