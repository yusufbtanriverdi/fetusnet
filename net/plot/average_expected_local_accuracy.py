import torch 
import matplotlib.pyplot as plt

def average_expected_local_accuracy(outputs, targets, spacings, radii):
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
    
    scaled_radii = [r / spacings[0] for r in radii]  # Scale radii by spacings[0]
    radii = [int(r / 2) for r in scaled_radii]      # Convert to integer radii
    num_samples, D, H, W = outputs.shape
    # Iterate over each radius
    distances = torch.zeros((num_samples, len(radii)), dtype=torch.float32)
    for n in range(num_samples):
        output, target = outputs[n], targets[n]  
        for ind, radius in enumerate(radii):
            landmark_coord = torch.nonzero(target == target.max(), as_tuple=False)[0]
            x, y, z = landmark_coord
            # Select neighbourhood
            mask = torch.zeros_like(output, dtype=output.dtype)
            mask[x-radius:x+radius, y-radius:y+radius, z-radius:z+radius] = 1
            roi = mask * output
            # Find candidate in ROI. 
            roi_peak = torch.nonzero(roi == roi.max(), as_tuple=False)[0]

            distance = torch.norm((roi_peak - landmark_coord).to(float))
            distances[n, ind] = distance

    edr = distances.mean(dim=0).tolist()
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(radii, edr, marker='o')
    plt.xlabel('Radius (mm)')   
    plt.ylabel('Average Expected Local Accuracy (AELA)')
    plt.grid()
    plt.savefig('average_expected_local_accuracy.png')
    plt.show()

    return edr