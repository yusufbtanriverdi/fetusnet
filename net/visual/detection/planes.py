import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def get_best_slices(ground_truth):
    """
    Find the slices (x, y, z) that contain the maximum landmark intensity in the ground truth heatmap.
    """
    max_index = np.unravel_index(np.argmax(ground_truth), ground_truth.shape)
    return max_index  # (z, y, x)

import numpy as np
import matplotlib.pyplot as plt

def overlay_heatmaps(us_image, gt_heatmap, pred_heatmap, alpha=0.5):
    """
    Generates overlay visualizations of the ultrasound image with the predicted and ground truth heatmaps separately.
    
    Args:
        us_image (numpy array): 3D ultrasound image (Z, Y, X)
        pred_heatmap (numpy array): 3D predicted heatmap (Z, Y, X)
        gt_heatmap (numpy array): 3D ground truth heatmap (Z, Y, X)
        alpha (float): Transparency level for overlays
    """
    # Get the most meaningful slice indices
    z_idx, y_idx, x_idx = get_best_slices(gt_heatmap)
    
    # Extract slices
    sagittal_img = us_image[:, :, x_idx]  # Sagittal plane (YZ)
    coronal_img = us_image[:, y_idx, :]  # Coronal plane (XZ)
    axial_img = us_image[z_idx, :, :]  # Axial plane (XY)
    
    sagittal_gt = gt_heatmap[:, :, x_idx]
    coronal_gt = gt_heatmap[:, y_idx, :]
    axial_gt = gt_heatmap[z_idx, :, :]
    
    sagittal_pred = pred_heatmap[:, :, x_idx]
    coronal_pred = pred_heatmap[:, y_idx, :]
    axial_pred = pred_heatmap[z_idx, :, :]
    
    # Create figure for ground truth overlays
    fig_gt, axes_gt = plt.subplots(1, 3, figsize=(15, 5))
    
    gt_planes = [
        (sagittal_img, sagittal_gt, 'Sagittal (YZ) - Ground Truth'),
        (coronal_img, coronal_gt, 'Coronal (XZ) - Ground Truth'),
        (axial_img, axial_gt, 'Axial (XY) - Ground Truth')
    ]
    
    for ax, (img, gt, title) in zip(axes_gt, gt_planes):
        ax.imshow(img, cmap='gray')  # Base ultrasound image
        ax.imshow(gt, cmap='Reds', alpha=alpha * (gt > 0))  # Overlay ground truth heatmap in red
        ax.set_title(title)
        ax.axis('off')
    
    # plt.show()
    
    # Create figure for predicted overlays
    fig_pred, axes_pred = plt.subplots(1, 3, figsize=(15, 5))
    
    pred_planes = [
        (sagittal_img, sagittal_pred, 'Sagittal (YZ) - Prediction'),
        (coronal_img, coronal_pred, 'Coronal (XZ) - Prediction'),
        (axial_img, axial_pred, 'Axial (XY) - Prediction')
    ]
    
    for ax, (img, pred, title) in zip(axes_pred, pred_planes):
        ax.imshow(img, cmap='gray')  # Base ultrasound image
        ax.imshow(pred, cmap='Blues', alpha=alpha * (pred > 0))  # Overlay predicted heatmap in blue
        ax.set_title(title)
        ax.axis('off')
    
    # plt.show()
    plt.close()
    
    return fig_gt, fig_pred  # Return both figure objects for saving or further processing
