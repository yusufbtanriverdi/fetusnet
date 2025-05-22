import torch 
import matplotlib.pyplot as plt

def plot_aela_figure(radii, edr, save_dir='average_expected_local_accuracy.png'):
    """
    Function to plot the average expected local accuracy (AELA) figure.
    This function is a placeholder and should be replaced with the actual implementation.
    """
    print("Plotting AELA figure...")
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
    plt.savefig(save_dir)
    # plt.show()
    # Save the plot as a PNG file
    plt.close()  # Close the plot to free up memory
    # Return the average expected local accuracy for each radius

def average_expected_local_accuracy(outputs, targets, spacings, radii, save_dir='average_expected_local_accuracy.png'):
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

    mm_radii = radii.clone()  # Store original radii for debugging
    scaled_radii = [r / 1.6 for r in radii]  # Scale radii by spacings[0]
    radii = [int(r / 2) for r in scaled_radii]      # Convert to integer radii
    print(f"Scaled Radii: {scaled_radii}, Integer Radii: {radii}")
    num_samples, D, H, W = outputs.shape
    # Iterate over each radius
    distances = torch.zeros((num_samples, len(radii)), dtype=torch.float32)

    ct = 0
    for n in range(num_samples):
        output, target = outputs[n], targets[n]  
        for ind, radius in enumerate(radii):
            landmark_coord = torch.nonzero(target == target.max(), as_tuple=False)[0]
            x, y, z = landmark_coord
            # Select neighbourhood
            if radius == 0:
                roi_peak = torch.tensor([x, y, z], dtype=torch.long)
            else:
                mask = torch.zeros_like(output, dtype=output.dtype)
                x_min, x_max = max(0, x-radius), min(D, x+radius)
                y_min, y_max = max(0, y-radius), min(H, y+radius)
                z_min, z_max = max(0, z-radius), min(W, z+radius)
                mask[x_min:x_max, y_min:y_max, z_min:z_max] = 1
                roi = mask * output
                if roi.max() == 0:
                    roi_peak = torch.tensor([x, y, z], dtype=torch.long)
                else:
                    roi_peak = torch.nonzero(roi == roi.max(), as_tuple=False)[0]
            # Calculate distance
            distance = torch.norm((roi_peak - landmark_coord).to(float))
            distances[n, ind] = distance
            # Print the results for debugging
            # Visualize the output 3D volume on the z-axis
            
            ct += 1
            # if landmark_coord[0] == 64 and landmark_coord[1] == 75 and landmark_coord[2] == 44:
            #     slice_z_output = output[:, :, z].cpu().numpy()
            #     slice_z_target = target[:, :, z].cpu().numpy()
            #     print(f"ROI Peak: {roi_peak}, Landmark Coord: {landmark_coord}, Radius: {radius}, Distance: {distance}")
                
            #     fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                
            #     # Output probability map
            #     axes[0].imshow(slice_z_output, cmap='gray')
            #     axes[0].scatter(roi_peak[1], roi_peak[0], color='orange', label='Candidate (x, y)')
            #     axes[0].scatter(y, x, color='red', label='Landmark (x, y)')
            #     axes[0].add_patch(plt.Rectangle((y-radius, x-radius), 2*radius, 2*radius, 
            #                                     edgecolor='blue', facecolor='none', linewidth=2, label='Bounding Box'))
            #     axes[0].set_title(f"Output Probability Map")
            #     axes[0].legend()
                
            #     # Target probability map
            #     axes[1].imshow(slice_z_target, cmap='gray')
            #     axes[1].scatter(roi_peak[1], roi_peak[0], color='orange', label='Candidate (x, y)')
            #     axes[1].scatter(y, x, color='red', label='Landmark (x, y)')
            #     axes[1].add_patch(plt.Rectangle((y-radius, x-radius), 2*radius, 2*radius, 
            #                                     edgecolor='blue', facecolor='none', linewidth=2, label='Bounding Box'))
            #     axes[1].set_title(f"Target Probability Map")
            #     axes[1].legend()
                
            #     plt.suptitle(f"Visualization at Z={z} with Radius={radius}")
            #     plt.tight_layout()
            #     # plt.close()
            #     plt.show()

    edr = distances.mean(dim=0)
    # Plotting the results
    plot_aela_figure(mm_radii, edr.tolist(), save_dir=save_dir)
    # Return the average expected local accuracy for each radius
    return edr