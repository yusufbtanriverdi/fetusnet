import torch
from net.model.modules.dsnt import dsnt_3d_separable
from net.plot.losses import imshow_target_distance_matrices_to_gif
from net.plot.histograms import plot_histograms_and_stats
from net.evaluation.similarity_scores import to_probability_distributions

def get_peak_location(heatmap, method):
    """Extracts peak location from a heatmap."""
    N_landmarks = heatmap.shape[0]  # Number of landmarks
    coords = torch.zeros((N_landmarks, 3), dtype=torch.float32)  # Initialize coordinates tensor
    if method not in ['argmax', 'com', 'dsnt']:    
        raise ValueError(f"Unsupported method: {method}. Use 'argmax', 'com' or 'dsnt'.")
    
    for i in range(N_landmarks):
        heatmap_i = heatmap[i]  # Extract the heatmap for the i-th landmark
        coords[i] = get_peak_location_single(heatmap_i, method)  # Get peak location for each landmark
    return coords  # Return the coordinates of the peaks for all landmarks

    
def get_peak_location_single(heatmap, method):
    """Extracts peak location from a single heatmap."""
    if method == 'argmax':
        peak = torch.nonzero(heatmap == heatmap.max(), as_tuple=False).float()
        return peak[0] if peak.numel() > 0 else None

    if method == 'com':
        D, H, W = heatmap.shape
        # Compute soft center-of-mass for smoother localization
        grid = torch.meshgrid(torch.arange(D, device=heatmap.device),
                              torch.arange(H, device=heatmap.device),
                              torch.arange(W, device=heatmap.device), indexing='ij')
        grid = torch.stack(grid, dim=-1).float()  # Shape (D, H, W, 3)
        # softmax over voxel domain: do reshape softmax for stability & correctness
        probs = torch.nn.functional.softmax(heatmap.view(-1), dim=-1).view(D, H, W)
        weighted = probs.unsqueeze(-1) * grid
        # imshow_target_distance_matrices_to_gif(weighted[:, :, :, 0], weighted[:, :, :, 1], weighted[:, :, :, 2], titles=['Weighted X', 'Weighted Y', 'Weighted Z'], gif_path='debug_com.gif')
        # print(heatmap.max().item(), heatmap.min().item(), heatmap.mean().item())
        # plot_histograms_and_stats(probs, weighted[:, :, :, 0])
        # Weighted sum of coordinates by probability
        com = weighted.sum(dim=(0, 1, 2))
        return com

    if method == 'dsnt':
        probs = to_probability_distributions(heatmap)
        coords = dsnt_3d_separable(probs, heatmap.device, heatmap.dtype)  # Use DSNT to get coordinates
        return coords