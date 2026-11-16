import numpy as np
import torch
import torch.nn.functional as F


def affine_grid_generator_3D(theta, size):
    """
    Generates a 3D affine transformation grid for resampling an image.

    Parameters:
        theta (torch.Tensor): Affine transformation matrix of shape (N, 3, 4).
        size (tuple): Shape of the output grid as (N, C, D, H, W).

    Returns:
        torch.Tensor: A 3D grid of shape (N, D, H, W, 3) for affine transformation.
    """
    N, _, D, H, W = size

    # Create an empty grid with an additional dimension for homogeneous coordinates
    base_grid = torch.zeros((N, D, H, W, 4), dtype=torch.float32)

    # Generate points in each dimension, normalized between -1 and 1
    w_points = torch.linspace(-1, 1, W) if D > 1 else torch.Tensor([-1])
    h_points = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])).unsqueeze(-1)
    d_points = (torch.linspace(-1, 1, D) if W > 1 else torch.Tensor([-1])).unsqueeze(-1).unsqueeze(-1)

    # Populate base grid with coordinates in each direction
    base_grid[:, :, :, :, 0] = w_points
    base_grid[:, :, :, :, 1] = h_points
    base_grid[:, :, :, :, 2] = d_points
    base_grid[:, :, :, :, 3] = 1  # Homogeneous coordinate

    # Apply the affine transformation to each point in the grid
    grid = torch.bmm(base_grid.view(N, D * H * W, 4), theta.transpose(1, 2))
    grid = grid.view(N, D, H, W, 3)  # Reshape to the target 3D grid shape

    return grid

def grid_transform_3d(moving_image, transform_matrix):
    """
    Applies an affine transformation to a 3D image.

    Parameters:
        moving_image (torch.Tensor): The input 3D image of shape (N, D, H, W).
        transform_matrix (torch.Tensor): Affine transformation matrix of shape (N, 3, 4).

    Returns:
        torch.Tensor: The transformed image.
    """
    # theta = np.linalg.inv(transform_matrix)
    theta = transform_matrix
    moving_image, theta = torch.from_numpy(moving_image).unsqueeze(0), torch.from_numpy(theta).unsqueeze(0)
    N, D, H, W = moving_image.shape
    # Add a channel dimension for grid_sample compatibility
    moving_image = moving_image.unsqueeze(1)
    # Generate the affine grid and apply it to the input image
    affine_grid = affine_grid_generator_3D(theta, (N, 1, D, H, W))
    predicted_image = F.grid_sample(moving_image, affine_grid)

    return predicted_image.squeeze(0).squeeze(0).numpy()  # Result after inverse mapping

def affine_transform(landmarks, R, T):
    """
    Applies an affine transformation to 3D landmarks.

    Parameters:
    - landmarks: A tensor of shape (3, N) containing the landmark coordinates.
    - theta: A tensor of shape (3, 4) representing the affine transformation matrix.

    Returns:
    - transformed_landmarks: A tensor of shape (3, N) with the transformed landmark coordinates.
    """

    landmarks = landmarks + T
    landmarks_transformed = np.dot(R, landmarks.T).T

    return landmarks_transformed

def affine3Dmatrix(gt):
    """
    Generates a 3D affine transformation matrix and quaternion based on given plane normals.

    Parameters:
        gt (dict): Dictionary containing 'sagittal', 'coronal', and 'axial' plane normals.

    Returns:
        tuple: (quaternion, matrix) where
            - matrix is a (3, 3) affine transformation matrix.
    """
    # Prepare normals for each plane, flipping the z-component for compatibility with LPS
    normal_sagittal = np.array([gt['sagittal'][0], gt['sagittal'][1], -gt['sagittal'][2]])
    normal_coronal = np.array([gt['coronal'][0], gt['coronal'][1], -gt['coronal'][2]])
    normal_axial = np.array([gt['axial'][0], gt['axial'][1], -gt['axial'][2]])

    # Default axis directions in 3D
    V0 = [[1, 0, 0], [0, 1, 0], [0, 0, -1]]
    V1 = [normal_sagittal, normal_axial, normal_coronal]
    # Solve for the affine rotation matrix
    M = np.transpose(np.linalg.solve(V0, V1))
    
    return M
