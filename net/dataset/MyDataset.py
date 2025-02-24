from net.dataset.target import gaussian_heatmap

import torchio as tio
from torch.utils.data import Dataset
import os 
import torch
import pandas as pd
import nrrd

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

    def __init__(self, dataframe, root, target_mode, target_params, lmk, transformations=None):
        """
        Initializes the dataset by loading 3D volumes and ground truth masks.

        Args:
            dataframe (pd.DataFrame): Dataframe containing scan metadata.
            root (str): Root directory containing the image and landmark files.
            target_mode (str): Target generation mode (only 'gaussian' is supported).
            target_params (tuple): Parameters (alpha, eps) for generating targets.
            lmk (str): Type of landmark to extract.
            transformations (callable, optional): Transformations to apply on volumes and masks.
        """
        self.dataframe = dataframe
        self.root = root
        self.target_mode = target_mode
        self.alpha, self.eps = target_params
        self.lmk = lmk
        self.transformations = transformations

    def __len__(self): 
        """Returns the number of 3D volumes in the dataset."""
        return len(self.dataframe)

    def __str__(self): 
        """Pompeu Fabra University @ 2025"""
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        """
        Retrieves a single 3D volume and its corresponding ground truth mask at the given index.

        Args:
            idx (int): Index of the volume to retrieve.

        Returns:
            tio.Subject: TorchIO Subject containing the image, target, and metadata.
        """
        # Load 3D volume
        image_path = os.path.join(self.root, self.dataframe.loc[idx, 'processed__vol_path'])
        volume, header = extract_image(image_path)
        image_tensor = torch.tensor(volume).unsqueeze(0)  # Add channel dimension

        # Load landmark file
        landmark_path = os.path.join(self.root, self.dataframe.loc[idx, 'processed__csv_path'])
        landmark_df = pd.read_csv(landmark_path)

        # Extract coordinates for the selected landmark
        landmark_row = landmark_df[landmark_df['label'] == self.lmk]
        if landmark_row.empty:
            raise ValueError(f"Landmark '{self.lmk}' not found in {landmark_path}")

        coord = landmark_row[['x', 'y', 'z']].iloc[0].tolist()  # Convert to list
        coord_tensor = torch.tensor(coord, dtype=torch.float32)

        # Generate target heatmap
        if self.target_mode == 'gaussian':
            target = torch.tensor(gaussian_heatmap.create_gaussian_heatmap(coord_tensor, volume, alpha=self.alpha, eps=self.eps)).unsqueeze(0)
        else:
            raise ValueError(f"Unsupported target mode: {self.target_mode}")

        if self.transformations is None:
            raise ValueError("Transformations are required but not provided.")

        # Apply transformations
        subject = tio.Subject(
            image=self.transformations(tio.ScalarImage(tensor=image_tensor)),  # Apply transformations
            target=tio.ScalarImage(tensor=target),
            name=self.dataframe.loc[idx, 'full_id'],
            pid=self.dataframe.loc[idx, 'pid'],
            spacings=header['spacings'][:3],
            coord_x=coord_tensor[0],
            coord_y=coord_tensor[1],
            coord_z=coord_tensor[2]
        )

        return subject
