import torch
import matplotlib.pyplot as plt
from net.postprocess.utility.where_is_landmark import get_peak_location


def plot_aela_figure(radii, edr, save_dir='average_expected_local_accuracy.png', show=False):
    """
    Plot the average expected local accuracy (AELA) figure.

    Args:
        radii (list or torch.Tensor): Radii in mm.
        edr (list or torch.Tensor): Average expected local accuracy values.
        save_dir (str): Path to save the figure.
    """
    upper_limit = [0.75 * r for r in radii]  # 3/4 upper limit

    plt.figure(figsize=(10, 6))
    plt.plot(radii, edr, label='Detector')
    plt.plot(radii, upper_limit, linestyle='--', color='blue', label='Upper Limit')
    plt.legend(loc='best')
    plt.xlabel('Radius (mm)')
    plt.ylabel('Average Expected Local Accuracy (AELA)')
    plt.yscale('log')
    plt.grid(True)

    if save_dir:
        plt.savefig(save_dir)

    if show:
        plt.show()

    plt.close()  # Free memory


def compute_aela(
    output,
    target_coord,
    spacing,
    radius_eval,
    radius_num,
    save_dir='curve.png',
    detector='argmax',
    show=False
):
    """
    Compute the average expected local accuracy (AELA) for a 3D output volume.

    Args:
        output (torch.Tensor): Model predictions (C, D, H, W), assuming one landmark per image.
        target_coord (tuple): Ground truth landmark coordinate (x, y, z).
        spacing (float): Voxel spacing in mm.
        radius_eval (float): Maximum radius for evaluation in mm.
        radius_num (int): Number of radii to evaluate.
        save_dir (str): Path to save the AELA curve figure.
        detector (str): Method to extract peak location ('argmax' or other supported).

    Returns:
        torch.Tensor: Average expected local accuracy for each radius.
    """
    # Define radii
    radii = torch.linspace(0, radius_eval, radius_num)
    mm_radii = radii.clone()  # Store original radii in mm
    vo_radii = [r / spacing for r in radii]  # Convert mm to voxel units
    radii = [int(r / 2) for r in vo_radii]  # Round to integer voxel radius

    # Initialize distances array
    distances = torch.zeros(len(radii), dtype=torch.float32)
    C, D, H, W = output.shape

    # Compute AELA for each radius
    for ind, radius in enumerate(radii):
        distances[ind] = 0

        if radius != 0:
            # Define bounding box around target

            x_min = torch.clamp(target_coord[0] - radius, min=0).round().int().item()
            x_max = torch.clamp(target_coord[0] + radius, max=D).round().int().item()
            y_min = torch.clamp(target_coord[1] - radius, min=0).round().int().item()
            y_max = torch.clamp(target_coord[1] + radius, max=H).round().int().item()
            z_min = torch.clamp(target_coord[2] - radius, min=0).round().int().item()
            z_max = torch.clamp(target_coord[2] + radius, max=W).round().int().item()

            # Mask region of interest
            mask = torch.zeros_like(output, dtype=torch.float32)
            mask[:, x_min:x_max, y_min:y_max, z_min:z_max] = 1
            roi = mask * output

            # Extract peak location in ROI
            output_coord = get_peak_location(roi, method=detector)

            # Calculate Euclidean distance
            distances[ind] = torch.norm((output_coord - target_coord).to(float))

    if show: 
        plot_aela_figure(mm_radii, distances.tolist(), save_dir=save_dir, show=True)

    return distances
