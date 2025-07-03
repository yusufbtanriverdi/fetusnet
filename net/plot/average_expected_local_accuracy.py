import torch 
import matplotlib.pyplot as plt
from net.postprocess.utility.where_is_landmark import get_peak_location

def plot_aela_figure(radii, edr, save_dir='average_expected_local_accuracy.png'):
    """
    Function to plot the average expected local accuracy (AELA) figure.
    This function is a placeholder and should be replaced with the actual implementation.
    """
    # print("Plotting AELA figure...")
    upper_limit = [3 / 4 * r for r in radii]
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(radii, edr, label='Detector')
    plt.plot(radii, upper_limit,  linestyle='--', color='blue', label='Upper Limit')
    plt.legend(loc = 'best')
    plt.xlabel('Radius (mm)')   
    plt.ylabel('Average Expected Local Accuracy (AELA)')
    plt.yscale('log')
    plt.grid()

    if save_dir:
        plt.savefig(save_dir)
    # plt.show()
    # Save the plot as a PNG file
    plt.close()  # Close the plot to free up memory
    # Return the average expected local accuracy for each radius

def average_expected_local_accuracy(output, target_coord, spacing, radius_eval, radius_num, save_dir='curve.png', landmark_extractor='argmax'):
    """
    Computes the average expected local accuracy (AELA) for a batch of 3D images.

    Args:
        outputs (torch.Tensor): Model predictions (batch_size, depth, height, width).
        targets (torch.Tensor): Ground truth labels (batch_size, depth, height, width).
        spacings (list): List of voxel spacings for each dimension.
        radii (list): List of radii for local accuracy computation.

    Returns:
        list: Average expected local accuracy for each channel.
    """
    radii = torch.linspace(0, radius_eval, radius_num)
    mm_radii = radii.clone()  # Store original radii for debugging
    vo_radii = [r / spacing for r in radii]  # Scale radii in mm by spacings to voxel radius
    radii = [int(r / 2) for r in vo_radii]      # Convert to integer radii
    
    # Iterate over each radius
    distances = torch.zeros((len(radii)), dtype=torch.float32)
    C, D, H, W = output.shape  # Assuming one landmark per image
    for ind, radius in enumerate(radii):
        distances[ind] = 0
        # Select neighbourhood
        if radius != 0:
            x_min, x_max = max(0, target_coord[0]-radius).round().int().item(), min(D, target_coord[0]+radius).round().int().item()
            y_min, y_max = max(0, target_coord[1]-radius).round().int().item(), min(H, target_coord[1]+radius).round().int().item()
            z_min, z_max = max(0, target_coord[2]-radius).round().int().item(), min(W, target_coord[2]+radius).round().int().item()
            mask = torch.zeros_like(output, dtype=torch.float32)
            mask[:, x_min:x_max, y_min:y_max, z_min:z_max] = 1
            roi = mask * output
            # Extract the peak location from the region of interest
            output_coord = get_peak_location(roi, method=landmark_extractor)
            # Calculate distance
            distances[ind]= torch.norm((output_coord - target_coord).to(float))
                
    # Plotting the results
    plot_aela_figure(mm_radii, distances.tolist(), save_dir=save_dir)
    # Return the average expected local accuracy for each radius
    return distances