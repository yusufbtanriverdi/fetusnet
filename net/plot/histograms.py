import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict


def plot_histograms_and_stats(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    bins: int = 50,
    save_path: Optional[str] = None,
    show: bool = True,
    range_percentile: float = 99.0,
    log_scale: bool = True,
    ) -> Dict[str, Dict[str, float]]:
    """
    Plot histograms of model outputs and targets with both full-range
    and zoomed-in views, and compute basic statistics.

    Args:
        outputs (torch.Tensor): Predicted logits.
        targets (torch.Tensor): Ground truth values.
        bins (int): Number of histogram bins. Default is 50.
        save_path (str, optional): If provided, saves the figure.
        show (bool): Whether to display the plot interactively.
        range_percentile (float): Percentile cutoff for zoomed-in histogram (default=99).
        log_scale (bool): If True, use log scale on y-axis.

    Returns:
        dict: Statistics (min, max, mean, std, sum) for both arrays.
    """
    outputs_np = outputs.detach().cpu().numpy().flatten()
    targets_np = targets.detach().cpu().numpy().flatten()

    def compute_stats(arr: np.ndarray) -> Dict[str, float]:
        return {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "sum": float(arr.sum()),
        }

    stats = {
        "outputs": compute_stats(outputs_np),
        "targets": compute_stats(targets_np),
    }

    def get_percentile_range(arr: np.ndarray):
        upper = np.percentile(arr, range_percentile)
        return (arr.min(), upper)

    # ----- Plot -----
    plt.figure(figsize=(14, 8))

    # Full-range Outputs
    plt.subplot(2, 2, 1)
    plt.hist(outputs_np, bins=bins, color="blue", alpha=0.7, label="Outputs (Full)")
    if log_scale:
        plt.yscale("log")
    plt.title("Outputs - Full Range")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()

    # Zoomed Outputs
    plt.subplot(2, 2, 2)
    plt.hist(outputs_np, bins=bins, range=get_percentile_range(outputs_np), color="blue", alpha=0.7, label=f"Outputs ≤ {range_percentile}th pct")
    if log_scale:
        plt.yscale("log")
    plt.title(f"Outputs - Zoomed (≤ {range_percentile}th pct)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()

    # Full-range Targets
    plt.subplot(2, 2, 3)
    plt.hist(targets_np, bins=bins, color="green", alpha=0.7, label="Targets (Full)")
    if log_scale:
        plt.yscale("log")
    plt.title("Targets - Full Range")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()

    # Zoomed Targets
    plt.subplot(2, 2, 4)
    plt.hist(targets_np, bins=bins, range=get_percentile_range(targets_np), color="green", alpha=0.7, label=f"Targets ≤ {range_percentile}th pct")
    if log_scale:
        plt.yscale("log")
    plt.title(f"Targets - Zoomed (≤ {range_percentile}th pct)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path.rstrip('/')}/histograms.png", dpi=300)

    if show:
        plt.show()

    plt.close()

    return stats

if __name__ == "__main__":
    # Dummy outputs: logits with a wide spread
    outputs = torch.randn(10_000) * 2  # Gaussian, std=2
    outputs[::500] = torch.randn(outputs[::500].shape) * 20  # Inject some big outliers

    # Dummy targets: mostly near 0 with occasional spikes
    targets = torch.rand(10_000) ** 3  # Skewed toward 0
    targets[::400] = torch.rand(targets[::400].shape) * 5  # Rare large values

    stats = plot_histograms_and_stats(
        outputs, targets,
        bins=50,
        range_percentile=99.0,
        log_scale=False,  # try True to see detail near 0
    )

    print("Stats:\n", stats)
