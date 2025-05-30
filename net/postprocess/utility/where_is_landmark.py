import torch

def get_peak_location(heatmap, method):
    """Extracts peak location from a heatmap."""
    if method == 'argmax':
        peak = torch.nonzero(heatmap == heatmap.max(), as_tuple=False).float()
        return peak[0] if peak.numel() > 0 else None

    elif method == 'com':
        # Compute soft center-of-mass for smoother localization
        coords = torch.meshgrid(torch.arange(heatmap.shape[0]),
                                torch.arange(heatmap.shape[1]),
                                torch.arange(heatmap.shape[2]), indexing='ij')
        coords = torch.stack(coords, dim=-1).float().to(heatmap.device)  # Shape (D, H, W, 3)

        heatmap = heatmap / heatmap.sum()  # Normalize to make it a probability map
        com = (heatmap.unsqueeze(-1) * coords).sum(dim=(0, 1, 2))  # Weighted sum
        return com
