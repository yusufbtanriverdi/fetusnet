import torch
import matplotlib.pyplot as plt

def create_histogram(tensor, n_bins):
    """
    Compute histograms for a batch of 3D images using torch.bincount.

    Args:
        tensors (torch.Tensor): Input tensor of shape (N, D, H, W).
        n_bins (int): Number of bins for the histogram.

    Returns:
        torch.Tensor: Histograms of shape (N, n_bins).
    """
    N = tensor.shape[0]  # Batch size
    device = tensor.device

    # Flatten each 3D image to a 1D vector in-place
    tensor = tensor.reshape(N, -1).detach().cpu()
    hist_1d = []
    # Initialize
    for i in range(N):
        img = tensor[i]
        # Compute the minimum and maximum values for the image
        min_val = 0
        max_val = 1

        if max_val == min_val:
            # If all pixel values are the same, assign all counts to bin 0.
            bin_idx = torch.zeros_like(img, dtype=torch.long) # could include device...
        else:
            # Scale each value into [0, n_bins)
            scaled = (img - min_val) / (max_val - min_val) * n_bins
            # Floor to get the bin index; convert to long
            bin_idx = scaled.floor().long()
            # Clamp indices to ensure they lie in the range [0, n_bins-1]
            bin_idx = torch.clamp(bin_idx, 0, n_bins - 1)

        # Count occurrences in each bin using torch.bincount.
        # Ensure that the output has exactly n_bins elements.
        counts = torch.bincount(bin_idx, minlength=n_bins).long().to(device)
        hist_1d.append(counts)
        import matplotlib.pyplot as plt
        # plt.bar(range(len(counts)), height=counts.cpu().numpy())
        # plt.show()
    
    return torch.stack(hist_1d, dim=0).to(float).to(device).requires_grad_(True)


# ==== Example Usage ====

# # Create a batch of 3D images with random values in [0, 1]
# N, D, H, W = 3, 5, 64, 64  # Batch of 3, Depth 5, Height 64, Width 64
# images = torch.rand(N, D, H, W)  # Random 3D images

if __name__ == '__main__':
    # Set number of histogram bins
    n_bins = 500

    import nrrd
    import numpy as np
    import seaborn as sns

    sns.set_style('white')

    def extract_image(filename):
        """Extract the image into a 3D numpy array [x, y, z]. As it was saved in RAS

        Args:
        filename: Path and name of nifti file.

        Returns:
        data: A 3D numpy array [x, y, z]
        pix_dim: pixel spacings

        """

        data, header = nrrd.read(filename)

        if len(data.shape) == 4:
            data=data[:, :, :, 0]
        
        return data, header
    

    hists = []
    for fname in ['runs/distance2025-03-12_22-03-34/10-30s-02.nrrd', 'runs/pixelwise_mse_otherfolds/10-30s-02.nrrd', '/home/yusuf/Source/fetusnet/runs/pixelwise_kld2025-03-10_11-12-59/10-30s-02.nrrd']:
        image, _ = extract_image(fname)
        image = torch.tensor(image).unsqueeze(0)
        # Compute histograms
        hist = create_histogram(image, n_bins)[0]
        hists.append(hist)
    
    histgt, histpr, histpr2 = hists[0].detach().cpu().numpy(), hists[1].detach().cpu().numpy(), hists[2].detach().cpu().numpy(), 
    width = 0.5
    # Create figure and primary axis (for bar chart)
    # Initialize a figure with a gridspec layout for marginal plots
    fig = plt.figure(figsize=(12, 6))

    # Scatter plot
    bar_ax = fig.add_subplot()
    # Bar chart (Histogram)
    bins = np.arange(1, n_bins+1) 
    bar_ax.set_xlim((0, 10))
    bar_ax.bar(bins, histgt, width=width, label="Ground Truth Histogram", alpha=0.7, color="lightblue")
    bar_ax.bar(bins + width, histpr, width=width, label="Predicted Histogram with Voxelwise MSE Loss", alpha=0.7, color="red")
    bar_ax.bar(bins + width*2, histpr2, width=width, label="Predicted Histogram with Voxelwise KLD Loss", alpha=0.7, color="purple")


    # Labels and legend
    bar_ax.set_xlabel("Bins")
    bar_ax.set_ylabel("Frequency (Histogram)")
    bar_ax.set_ylabel("Density (KDE)")

    bar_ax.legend(loc="upper right")
    plt.title("Histogram with KDE Overlay")

    # KDE plot
    # Marginal histogram (top)
    fig, ax= plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.set_xlim((-0.5* 1e6, 1e6))
    sns.kdeplot(histgt, fill=True, alpha=0.5, color="lightblue",ax=ax, linewidth=3, label="KDE of Ground Truth")
    sns.kdeplot(histpr, fill=True, alpha=0.5, color="red", ax=ax, linewidth=3, label="KDE of Predictions")
    sns.kdeplot(histpr2, fill=True, alpha=0.5, color="purple", ax=ax, linewidth=3, label="KDE of Predictions")

    # Create a secondary y-axis for the KDE plot
    # Legends for both axes
    ax.legend(loc="upper right")

    plt.show()