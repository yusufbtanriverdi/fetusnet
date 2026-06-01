import torch
from net.model.modules.dsnt import dsnt_3d_separable
from net.plot.matrices_3d import plot_3d_matrices
from torch.nn import functional as F

def _softmax(volume):
    C, D, H, W = volume.shape
    # Apply softmax over voxel domain.
    voxels = volume.reshape(C, -1) # Reshape to voxels
    probs = F.softmax(voxels, dim=-1).view(C, D, H, W)
    return probs

def _sigmoid(volume):
    return F.sigmoid(volume)

def get_peak_location(heatmap, method, **args):
    """Extracts peak location from a heatmap."""
    N_landmarks = heatmap.shape[0]  # Number of landmarks
    coords = torch.zeros((N_landmarks, 3), dtype=torch.float32)  # Initialize coordinates tensor
    if method not in ['argmax', 'com', 'dsnt']:    
        raise ValueError(f"Unsupported method: {method}. Use 'argmax', 'com' or 'dsnt'.")
    probs = _sigmoid(heatmap)  # Convert to probability distribution.
    for i in range(N_landmarks):
        coords[i] = get_peak_location_single(probs[i], method, mean_multi_peak=args.get('mean_multi_peak', True))  # Get peak location for each landmark
    return coords  # Return the coordinates of the peaks for all landmarks
    
def get_peak_location_single(probs, method, mean_multi_peak=True):
    """Extracts peak location from a single heatmap."""
    if method == 'argmax':
        # Find all peak locations (handles ties) and return their mean as a single float coordinate
        max_val = probs.max()
        peaks = torch.nonzero(probs == max_val, as_tuple=False).float()
        if peaks.numel() == 0:
            # fallback: return zeros if something unexpected happens
            raise ValueError("No peaks found in the heatmap. Check the input probabilities.")
        if peaks.shape[0] > 1:
            # If multiple voxels share the max value, average their coordinates for a stable center
            # print(f"Warning: Multiple peaks found with the same max value.")
            if mean_multi_peak:
                # print("Averaging their coordinates for stability \b.", peaks.shape, peaks[0])
                return peaks.mean(dim=0)
            else:                
                # print("Returning the first peak found \b.", peaks.shape, peaks[0])
                return peaks[0].float()
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