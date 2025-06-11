from net.dataset.target import gaussian_heatmap
from net.dataset.target import distance_matrix
import torchio as tio
from torch.utils.data import Dataset
import os 
import torch
import pandas as pd
import nrrd
import numpy as np
# Import necessary libraries
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

class MyDataset(Dataset):
    """
    Custom dataset class for handling 3D NIfTI images and their ground truth (GT) heatmaps.
    """

    def __init__(self, dataframe, root, target_mode, target_params, lmks, transformations=None):
        """
        Initializes the MyDataset class for loading and processing 3D medical imaging data.

        Args:
            dataframe (pd.DataFrame): A pandas DataFrame containing metadata for the scans, 
            such as file paths and associated labels.
            root (str): The root directory where the image files and landmark files are stored.
            target_mode (str): The mode for generating target outputs. Supported modes are 
            'gaussian' and 'distance' for generating heatmaps or similar targets.
            target_params (tuple): A tuple containing parameters (alpha, eps) used for generating 
            the target outputs. 'alpha' controls the spread of the target distribution, 
            and 'eps' is a small value for numerical stability.
            lmk (str): The type or name of the landmark to extract from the data 
            (e.g., anatomical points).
            transformations (callable, optional): A callable object or function to apply data 
            augmentations or preprocessing on the input volumes and masks. Defaults to None.
            return_dist_m (bool, optional): If True, the dataset will also return the distance map 
            along with the processed data. Defaults to False.
        """
        # Validate inputs
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame.")
        if not os.path.isdir(root):
            raise ValueError(f"Root directory '{root}' does not exist.")
        if target_mode not in ['gaussian', 'distance']:
            raise ValueError("target_mode must be either 'gaussian' or 'distance'.")
        if not isinstance(target_params, tuple) or len(target_params) != 2:
            raise ValueError("target_params must be a tuple of length 2 (alpha, eps).")
        if not isinstance(lmks, list) or not all(isinstance(lmk, str) for lmk in lmks):
            raise TypeError("lmk must be a list of strings.")
        if transformations is not None and not callable(transformations):
            raise TypeError("transformations must be callable or None.")
        
        self.dataframe = dataframe
        self.root = root
        self.target_mode = target_mode
        self.alpha, self.eps = target_params
        self.lmks = lmks
        self.transformations = transformations

    def __len__(self): 
        """Returns the number of 3D volumes in the dataset."""
        return len(self.dataframe)

    def __str__(self): 
        """
        Created by Yusuf B. Tanrıverdi, 2025 @ Pompeu Fabra University (UPF).

        If Karl Marx were here, he'd probably say:
        'The dataset is the opium of the programmer.'
        """
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        Retrieves a single 3D volume and its corresponding ground truth mask at the given index.

        Args:
            idx (int): Index of the volume to retrieve.

        Returns:
            tio.Subject: TorchIO Subject containing the image, target, and metadata.
        """
        # Load the 3D volume from the specified path
        image_path = os.path.join(self.root, self.dataframe.loc[idx, 'processed__vol_path'])
        volume, header = extract_image(image_path)
        image_tensor = torch.tensor(volume).unsqueeze(0)  # Add channel dimension (C, H, W, D)

        # Load the landmark file containing coordinates
        landmark_path = os.path.join(self.root, self.dataframe.loc[idx, 'processed__csv_path'])
        landmark_df = pd.read_csv(landmark_path)

        target = torch.zeros((len(self.lmks), *volume.shape), dtype=torch.float32)  # Initialize target tensor
        coord_tensor = torch.zeros((len(self.lmks), 3), dtype=torch.float32)  # Initialize coordinates tensor
        # Extract the coordinates for the specified landmark
        for i, lmk in enumerate(self.lmks):
            if lmk not in landmark_df['label'].values:
                raise ValueError(f"Landmark '{lmk}' not found in {landmark_path}")

            landmark_row = landmark_df[landmark_df['label'] == lmk]
            if landmark_row.empty:
                raise ValueError(f"Landmark '{lmk}' not found in {landmark_path}")

            # Convert landmark coordinates to a tensor and adjust for pixel spacing
            coord = landmark_row[['x', 'y', 'z']].iloc[0].tolist()
            coord = coord / header['spacings'][:3]  # Normalize by pixel spacings
            coord = torch.tensor(coord, dtype=torch.float32)
            coord_tensor[i] = coord  # Store the coordinates in the tensor
            # Coords are in voxel coordinates, now. 

            # Generate the target output based on the specified mode (gaussian or distance)
            if self.target_mode == 'gaussian':
                target_ = torch.tensor(
                    gaussian_heatmap.create_gaussian_heatmap(
                        coord, volume, alpha=self.alpha, eps=self.eps
                    )
                ).unsqueeze(0)  # Add channel dimension
            elif self.target_mode == 'distance':
                target_ = torch.tensor(
                    distance_matrix.create_distance_matrix(
                        coord, volume, alpha=self.alpha, eps=self.eps
                    )
                ).unsqueeze(0)  # Add channel dimension
            else:
                raise ValueError(f"Unsupported target mode: {self.target_mode}")

            target[i] = target_  # Assign the generated target to the corresponding landmark index


        # Ensure transformations are provided
        if self.transformations is None:
            raise ValueError("Transformations are required but not provided.")

        # Apply transformations and create a TorchIO Subject
        subject = tio.Subject(
            image=self.transformations(tio.ScalarImage(tensor=image_tensor)),  # Apply transformations to the image
            target=tio.ScalarImage(tensor=target),  # Target heatmap or distance map
            name=self.dataframe.loc[idx, 'full_id'],  # Metadata: full ID of the subject
            pid=self.dataframe.loc[idx, 'pid'],  # Metadata: patient ID
            spacings=header['spacings'][:3],  # Pixel spacings
            coords = coord_tensor,  # Coordinates of landmarks
            lmk=self.lmks  # List of landmarks
        )

        return subject
