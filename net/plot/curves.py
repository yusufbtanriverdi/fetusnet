import torch
import matplotlib.pyplot as plt
from net.model.modules.dsnt import dsnt_3d_separable
from net.postprocess.utility.where_is_landmark import get_peak_location


# def get_peak_location_single_v2(probs, target_coords, method):
#     """Extracts peak location from a single heatmap."""
#     if method == 'argmax':
#         # Find all peak locations (handles ties) and return the one closest to the target coordinate
#         max_val = probs.max()
#         peaks = torch.nonzero(probs == max_val, as_tuple=False).float()
#         if peaks.numel() == 0:
#             raise ValueError("No peaks found in the heatmap. Check the input probabilities.")
#         if target_coords is not None:
#             target = torch.tensor(target_coords, device=peaks.device, dtype=peaks.dtype)
#             distances = torch.norm(peaks - target.unsqueeze(0), dim=1)
#             return peaks[distances.argmin()]
#         return peaks[0]

#     if method == 'com':
#         D, H, W = probs.shape
#         # Compute soft center-of-mass for smoother localization
#         grid = torch.meshgrid(torch.arange(D, device=probs.device),
#                               torch.arange(H, device=probs.device),
#                               torch.arange(W, device=probs.device), indexing='ij')
#         grid = torch.stack(grid, dim=-1).float()  # Shape (D, H, W, 3)
#         # softmax over voxel domain: do reshape softmax for stability & correctness
#         weighted = probs.unsqueeze(-1) * grid
#         # plot_3d_matrices(probs, weighted[: ,: ,: , 0], weighted[:, :, :, 1], weighted[:, :, :, 2], titles=['Probabilities', 'Weighted X', 'Weighted Y', 'Weighted Z'])
#         # Weighted sum of coordinates by probability
#         com = weighted.sum(dim=(0, 1, 2))
#         return com

#     if method == 'dsnt':
#         # probs = to_probability_distributions(heatmap)
#         coords = dsnt_3d_separable(probs, probs.device, probs.dtype)  # Use DSNT to get coordinates
#         return coords

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
    distance_map,
    spacing,
    radii,
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
    mm_radii = radii.clone()  # Store original radii in mm
    vo_radii = [r / spacing for r in radii]  # Convert mm to voxel units
    radii = [int(r / 2) for r in vo_radii]  # Round to integer voxel radius

    # Initialize distances array
    distances = torch.zeros(len(radii), dtype=torch.float32)
    _, D, H, W = output.shape

    # Compute AELA for each radius
    for ind, radius in enumerate(radii[:-1]):
        if radius != 0:
            # Mask region of interest
            mask = (distance_map <= radius).float()
            roi = mask * output
            # Extract peak location in ROI
            output_coord = get_peak_location(roi, method=detector, eval=True, target_coord=target_coord)
            if roi[:, output_coord[0][0].int(), output_coord[0][1].int(), output_coord[0][2].int()] == 0:
                # If the predicted coordinate is outside the mask (i.e., no valid prediction), set distance to radius
                distances[ind+1] = radius * spacing  # Convert back to mm for distance
                print("No valid prediction within radius. Setting distance to radius: ", distances[ind])
                continue
            # Calculate Euclidean distance
            distances[ind+1] = torch.norm((output_coord - target_coord).to(float))
            # if lmk == 'enR':
            #     print("Distance" , distances[ind], "Predicted: ", output_coord, "Target: ", target_coord, "Radius: ", radius, "Mask nonzero radius: ", mask.nonzero().shape)
            if distances[ind+1] < distances[ind]:
                print("Warning! Curren  distance is lower than previous one, which shouldnt be possible!")
                print(f"Distance from radius {radius}: {distances[ind]} between predicted coord {output_coord} and target {target_coord}")            
                print(f"Distance from radius {radii[ind-1]}: {distances[ind+1]} between predicted coord {output_coord} and target {target_coord}")            
    if show: 
        plot_aela_figure(mm_radii, distances.tolist(), save_dir=save_dir, show=True)

    return distances
