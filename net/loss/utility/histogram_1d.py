import torch
import matplotlib.pyplot as plt

def create_histogram(tensors, n_bins):
    """
    Compute histograms for a batch of 3D images.

    Args:
        tensors (torch.Tensor): Input tensor of shape (N, D, H, W).
        n_bins (int): Number of bins for the histogram.

    Returns:
        torch.Tensor: Histograms of shape (N, n_bins).
    """
    N = tensors.shape[0]  # Batch size
    histograms = []

    for i in range(N):
        flattened = tensors[i].flatten()  # Flatten to 1D
        hist = torch.histc(flattened, bins=n_bins, min=flattened.min().item(), max=flattened.max().item())
        histograms.append(hist)

    return torch.stack(histograms)  # Shape: (N, n_bins)

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
