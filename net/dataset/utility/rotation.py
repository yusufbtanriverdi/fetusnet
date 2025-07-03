import torch
import torch.nn.functional as F
import nrrd
import numpy as np
from os.path import join
from scipy import ndimage

def get_file_list(txt_file):
    """Get a list of filenames.

    Args:
      txt_file: Name of a txt file containing a list of filenames for the images.

    Returns:
      filenames: A list of filenames for the images.

    """
    with open(txt_file) as f:
        filenames = f.read().splitlines()
    return filenames

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


def affine_grid_generator_3D(theta, size):
    """
    Generates a 3D affine transformation grid for resampling an image.

    Parameters:
        theta (torch.Tensor): Affine transformation matrix of shape (N, 3, 4).
        size (tuple): Shape of the output grid as (N, C, D, H, W).

    Returns:
        torch.Tensor: A 3D grid of shape (N, D, H, W, 3) for affine transformation.
    """
    N, C, D, H, W = size

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

def filter_3d_image(volume, filter_size=3, mode='reflect'):

    low_pass_filter = np.ones((filter_size,) * 3) / (filter_size ** 3)

    # Apply filtering for smoothing
    return ndimage.convolve(volume, low_pass_filter, mode=mode)

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

def affine_transform_3d(image, transform_matrix):
    depth, height, width = image.shape
    x, y, z = np.meshgrid(np.arange(width), np.arange(height), np.arange(depth))
    homogeneous_coords = np.stack([x, y, z, np.ones_like(x)], axis=-1).reshape(-1, 4).T
    
    transformed_coords = np.dot(transform_matrix, homogeneous_coords)
    new_x, new_y, new_z, _ = transformed_coords.reshape(4, depth, height, width)
    
    transformed_image = ndimage.map_coordinates(image, [new_z, new_y, new_x], order=1, mode='constant')
    return transformed_image


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

def translate_lmk(landmarks, translation_vector):
    return landmarks + translation_vector

def get_matrix_of_lmks(landmarks_df):
    """
    Convert the DataFrame of landmarks into a matrix in LPS (Left-Posterior-Superior) format.
    
    Parameters:
    landmarks_df (pd.DataFrame): DataFrame containing landmark data.
    
    Returns:
    np.ndarray: Numpy array representing the matrix of landmarks in LPS format.
    """

    # Assuming the DataFrame has columns named 'x', 'y', 'z' for the coordinates
    # Modify according to your actual column names
    # x = -landmarks_df['x'].values  # Flip x for LPS
    # y = -landmarks_df['y'].values  # Flip y for LPS
    # z = landmarks_df['z'].values

    # There are some instances that is in RAS but do not need this sign change. Maybe abs is better?
    x = abs(landmarks_df['x'].values)  # Flip x for LPS
    y = abs(landmarks_df['y'].values)  # Flip y for LPS
    z = abs(landmarks_df['z'].values)

    # This works but still the case 39-20s-08 is shifted a bit. 
    coords = np.array([x, y, z])
    lmk_matrix = np.column_stack(coords).astype(np.float32)
    
    return lmk_matrix

def swap_xz_coordinates(lmks):
    """
    Swaps the x and z coordinates of each landmark point in the given array.

    Args:
        lmks (numpy.ndarray or torch.Tensor): A 2D array of landmarks with shape (N, 3), 
                                              where N is the number of landmarks, and the 
                                              coordinates are [x, y, z].

    Returns:
        numpy.ndarray or torch.Tensor: The array with x and z coordinates swapped.
    """
    lmks[:, [0, 2]] = lmks[:, [2, 0]]
    return lmks


def pad_3d_image(image, lmks, centers_gt, desired_size):
    pad_width = [(max(d - s, 0) // 2, max(d - s, 0) - max(d - s, 0) // 2) for s, d in zip(image.shape, desired_size)]
    padded_image = np.pad(image, pad_width, mode='constant')
    
    # Adjust landmarks
    pad_before = np.array([p[0] for p in pad_width])
    adjusted_lmks = lmks + pad_before
    adjusted_centers_gt = centers_gt + pad_before
    
    return padded_image, adjusted_lmks, adjusted_centers_gt


def save_transformed_landmarks_arc(Lhat, lmk, output_path, name, prefix='tr'):
    """
    Save transformed landmarks to CSV and FCSV files with predefined header lines.

    Parameters:
    - Lhat (np.ndarray): Array containing transformed landmark coordinates.
    - lmk (pd.DataFrame): DataFrame to copy structure from.
    - output_path (str): Directory path to save the files.
    - name (str): Base name for the output files.
    """
    # Copy structure of landmarks and update with transformed coordinates
    Lhat_df = lmk.copy()
    for idx in range(Lhat.shape[0]):
        Lhat_df.loc[idx, ['x', 'y', 'z']] = Lhat[idx, :]

    # Save CSV file
    csv_path = f'{output_path}/csv/{prefix}{name}.csv'
    Lhat_df.to_csv(csv_path, index=False)

    # Define the predefined lines for FCSV format
    predefined_lines = [
        "# Markups fiducial file version = 5.2\n",
        "# CoordinateSystem = LPS\n",
        "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n"
    ]

    # Save FCSV file with predefined lines
    fcsv_path = f'{output_path}/fcsv/{prefix}{name}.fcsv'
    with open(fcsv_path, 'w') as f:
        # Write predefined lines
        f.writelines(predefined_lines)
        
        # Write each row as a comma-separated string
        for index, row in Lhat_df.iterrows():
            f.write(','.join(map(str, row.values)) + '\n')


def save_3d_image_arc(Vhat, header, desired_size, output_path, name, prefix=''):
    # Save padded 3D image as .nrrd
    header_ = header.copy()
    header_['sizes'] = desired_size  # Update to the desired size

    nrrd.write(join(output_path, f"{prefix}{name}.nrrd"), Vhat, header=header_)

def save_3d_image(Vhat, header, desired_size, output_path):
    # Save padded 3D image as .nrrd
    header_ = header.copy()
    header_['sizes'] = desired_size  # Update to the desired size

    nrrd.write(output_path, Vhat, header=header_)

def save_transformed_landmarks(Lhat, lmk, output_csv_path, output_fscv_path):
    """
    Save transformed landmarks to CSV and FCSV files with predefined header lines.

    Parameters:
    - Lhat (np.ndarray): Array containing transformed landmark coordinates.
    - lmk (pd.DataFrame): DataFrame to copy structure from.
    - output_path (str): Directory path to save the files.
    - name (str): Base name for the output files.
    """
    # Copy structure of landmarks and update with transformed coordinates
    Lhat_df = lmk.copy()
    for idx in range(Lhat.shape[0]):
        Lhat_df.loc[idx, ['x', 'y', 'z']] = Lhat[idx, :]

    # Save CSV file
    Lhat_df.to_csv(output_csv_path, index=False)

    # Define the predefined lines for FCSV format
    predefined_lines = [
        "# Markups fiducial file version = 5.2\n",
        "# CoordinateSystem = LPS\n",
        "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n"
    ]

    # Save FCSV file with predefined lines
    fcsv_path = output_fscv_path
    with open(fcsv_path, 'w') as f:
        # Write predefined lines
        f.writelines(predefined_lines)
        
        # Write each row as a comma-separated string
        for index, row in Lhat_df.iterrows():
            f.write(','.join(map(str, row.values)) + '\n')