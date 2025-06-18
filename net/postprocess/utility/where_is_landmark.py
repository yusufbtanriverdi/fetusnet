import torch

def get_peak_location(heatmap, method):
    """Extracts peak location from a heatmap."""
    N_landmarks = heatmap.shape[0]  # Number of landmarks
    coords = torch.zeros((N_landmarks, 3), dtype=torch.float32)  # Initialize coordinates tensor
    if method not in ['argmax', 'com']:    
        raise ValueError(f"Unsupported method: {method}. Use 'argmax' or 'com'.")
    
    for i in range(N_landmarks):
        heatmap_i = heatmap[i]  # Extract the heatmap for the i-th landmark
        coords[i] = get_peak_location_single(heatmap_i, method)  # Get peak location for each landmark
    return coords  # Return the coordinates of the peaks for all landmarks

    
def get_peak_location_single(heatmap, method):
    """Extracts peak location from a single heatmap."""
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
