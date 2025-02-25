import torch
import bisect

## MANUAL ## 
def find_bin(sorted_arr, num):
    index = bisect.bisect_left(sorted_arr, num)  # Find the insertion index
    left_idx = index - 1 if index > 0 else 0  # Left boundary (None if it's the first position)
    right_idx = index if index < len(sorted_arr) else len(sorted_arr)  # Right boundary (None if it's after the last element)
    # print(sorted_arr, num, left_idx, right_idx)
    return left_idx

def create_joint_histogram(x, y, n_bins):
    """
    Creates a joint histogram of two 3D tensors.
    
    Args:
        x (torch.Tensor): First 3D tensor of shape (D, H, W).
        y (torch.Tensor): Second 3D tensor of shape (D, H, W).
        n_bins (int): Number of bins for histogram.

    Returns:
        torch.Tensor: Joint histogram of shape (n_bins, n_bins).
    """

    assert x.shape == y.shape, "Input tensors must have the same shape"

    # Get min and max for proper binning
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())

    # Compute bin edges like torch.histogramdd()
    bin_edges = torch.linspace(min_val, max_val, n_bins + 1)
    histogram = torch.zeros((n_bins, n_bins))

    #x_flat, y_flat = ((x - min_val) / (max_val - min_val)).flatten() * n_bins, ((y - min_val) / (max_val - min_val)).flatten() * n_bins
    # Populate histogram
    for i in range(x.numel()):
        bin_x = find_bin(bin_edges.tolist(), x.flatten()[i].item())
        bin_y = find_bin(bin_edges.tolist(), y.flatten()[i].item())
        # print("Coord: ", i, "Intensity values: ", (x.flatten()[i].item(), y.flatten()[i].item()), "Bins assigned: ", (bin_x, bin_y))

        histogram[bin_x, bin_y] += 1

    return histogram

## PYTORCH ##

def create_joint_histogram_fast(x, y, n_bins):
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # Get min and max for proper binning
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())

    hist, _ = torch.histogramdd(torch.stack((x_flat, y_flat), dim=1).to(torch.float32), bins=(n_bins, n_bins), 
                                range=[float(min_val), float(max_val), float(min_val), float(max_val)])
    return hist

