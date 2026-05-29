import torch
from net.model.modules.dsnt import dsnt_3d_separable
from net.evaluation.similarity_scores import to_probability_distributions
from net.plot.matrices_3d import plot_3d_matrices

def get_peak_location(heatmap, method):
    """Extracts peak location from a heatmap."""
    N_landmarks = heatmap.shape[0]  # Number of landmarks
    coords = torch.zeros((N_landmarks, 3), dtype=torch.float32)  # Initialize coordinates tensor
    if method not in ['argmax', 'com', 'dsnt']:    
        raise ValueError(f"Unsupported method: {method}. Use 'argmax', 'com' or 'dsnt'.")
    
    for i in range(N_landmarks):
        heatmap_i = heatmap[i]  # Extract the heatmap for the i-th landmark
        probs = to_probability_distributions(heatmap_i)  # Convert to probability distribution {?} Why?
        coords[i] = get_peak_location_single(probs, method)  # Get peak location for each landmark
    return coords  # Return the coordinates of the peaks for all landmarks

    
def get_peak_location_single(probs, method):
    """Extracts peak location from a single heatmap."""
    if method == 'argmax':
        # Find all peak locations (handles ties) and return their mean as a single float coordinate
        max_val = probs.max()
        peaks = torch.nonzero(probs == max_val, as_tuple=False).float()
        if peaks.numel() == 0:
            # fallback: return zeros if something unexpected happens
            raise ValueError("No peaks found in the heatmap. Check the input probabilities.")
        # If multiple voxels share the max value, average their coordinates for a stable center
        return peaks[0].float()

    if method == 'com':
        D, H, W = probs.shape
        # Compute soft center-of-mass for smoother localization
        grid = torch.meshgrid(torch.arange(D, device=probs.device),
                              torch.arange(H, device=probs.device),
                              torch.arange(W, device=probs.device), indexing='ij')
        grid = torch.stack(grid, dim=-1).float()  # Shape (D, H, W, 3)
        # softmax over voxel domain: do reshape softmax for stability & correctness
        weighted = probs.unsqueeze(-1) * grid
        # plot_3d_matrices(probs, weighted[: ,: ,: , 0], weighted[:, :, :, 1], weighted[:, :, :, 2], titles=['Probabilities', 'Weighted X', 'Weighted Y', 'Weighted Z'])
        # Weighted sum of coordinates by probability
        com = weighted.sum(dim=(0, 1, 2))
        return com

    if method == 'dsnt':
        # probs = to_probability_distributions(heatmap)
        coords = dsnt_3d_separable(probs, probs.device, probs.dtype)  # Use DSNT to get coordinates
        return coords