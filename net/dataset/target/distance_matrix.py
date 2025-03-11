import numpy as np


def create_distance_matrix(coord, template, alpha = 3, eps=1e-6, **args):

    """
    Compute the Gaussian distribution value based on distance from a landmark.

    Args:
    coord : array-like
        Landmark coordinates, i.e., the mean of the Gaussian.
    template : np.ndarray volume that corresponds to this gaussian heatmap. 
    alpha : float
        Coefficient controlling the spread of the Gaussian.

    Returns:
    heatmap : a 3D matrix same shape as _template.shape_
        Gaussian value at pixel x relative to the landmark.
    """
    # Convert from torch to numpy array.
    coord = coord.numpy()
    # Unpack the input dimensions
    D, H, W = template.shape
    # Create a grid of all possible (d, h, w) pixel coordinates
    # d_range: [0, 1, ..., D-1]
    # h_range: [0, 1, ..., H-1]
    # w_range: [0, 1, ..., W-1]
    d_range = np.arange(D)
    h_range = np.arange(H)
    w_range = np.arange(W)
    
    # Use np.meshgrid to create a coordinate grid for the entire heatmap
    # d_grid, h_grid, w_grid each have shape (D, H, W) and represent all possible coordinates in 3D space
    d_grid, h_grid, w_grid = np.meshgrid(d_range, h_range, w_range, indexing='ij')
    # Calculate the squared Euclidean distance for every coordinate in the grid relative to the landmark
    distance = np.sqrt((d_grid - coord[0])**2 + (h_grid - coord[1])**2 + (w_grid - coord[2])**2)

    heatmap = 1 - np.exp(-distance / (2 * alpha**2)) 

    # Clip very low values ( will create a zero region around landmark)
    heatmap[heatmap < eps] = 0

    # Find indices where `template` is 0 and set corresponding heatmap values to 0
    heatmap[template == 0] = 0

    return heatmap