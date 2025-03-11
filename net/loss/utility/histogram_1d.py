import torch
import matplotlib.pyplot as plt

def create_histogram(tensors, n_bins):
    """
    Compute histograms for a batch of 3D images using torch.bincount.

    Args:
        tensors (torch.Tensor): Input tensor of shape (N, D, H, W).
        n_bins (int): Number of bins for the histogram.

    Returns:
        torch.Tensor: Histograms of shape (N, n_bins).
    """
    N = tensors.shape[0]  # Batch size
    device = tensors.device

    # Flatten each 3D image to a 1D vector
    flattened = tensors.view(N, -1)

    histos = torch.zeros((N, n_bins), device=device)
    for i in range(N):
        img = flattened[i]
        # Compute the minimum and maximum values for the image
        min_val = img.min()
        max_val = img.max()

        if max_val == min_val:
            # If all pixel values are the same, assign all counts to bin 0.
            bin_idx = torch.zeros_like(img, dtype=torch.long)
        else:
            # Scale each value into [0, n_bins)
            scaled = (img - min_val) / (max_val - min_val) * n_bins
            # Floor to get the bin index; convert to long
            bin_idx = scaled.floor().long()
            # Clamp indices to ensure they lie in the range [0, n_bins-1]
            bin_idx = torch.clamp(bin_idx, 0, n_bins - 1)

        # Count occurrences in each bin using torch.bincount.
        # Ensure that the output has exactly n_bins elements.
        count = torch.bincount(bin_idx, minlength=n_bins).to(device)
        histos[i] = count


        plt.bar(range(len(histos[i])), height=histos[i].cpu().numpy())
        plt.show()

    histos.requires_grad = True
    return histos  # Shape: (N, n_bins)


def do_something(tensor, *args):
    tensor = torch.zeros_like(tensor, device=tensor.device, requires_grad=True)
    return tensor

# ==== Example Usage ====

# # Create a batch of 3D images with random values in [0, 1]
# N, D, H, W = 3, 5, 64, 64  # Batch of 3, Depth 5, Height 64, Width 64
# images = torch.rand(N, D, H, W)  # Random 3D images

# # Set number of histogram bins
# n_bins = 20

# # Compute histograms
# histograms = create_histogram(images, n_bins)

# # Print the shape of the histogram output
# print("Histogram shape:", histograms.shape)  # Expected: (3, 20)

# # ==== Plot Example Histogram ====

# # Convert to numpy for visualization
# hist_np = histograms[0].numpy()  # Histogram of the first image

# plt.bar(range(n_bins), hist_np)
# plt.xlabel("Bins")
# plt.ylabel("Frequency")
# plt.title("Histogram of First 3D Image")
# plt.show()
