from dataset.target import gaussian_heatmap
from dataset.utility.rotation import extract_image
from net.plot.heatmaps import plot_heatmaps_slices_from_coord
import torch
import os 
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torchio as tio


class MyDataset(Dataset):
    """
    Custom dataset class for handling 3D NIfTI images and their ground truth (GT) heatmaps.
    """

    def __init__(self, dataframe, root, target_params, lmks, transformations=None):
        """
        Initializes the MyDataset class for loading and processing 3D medical imaging data.

        Args:
            dataframe (pd.DataFrame): A pandas DataFrame containing metadata for the scans, 
            such as file paths and associated labels.
            root (str): The root directory where the image files and landmark files are stored.
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
        if not isinstance(target_params, tuple) or len(target_params) != 4:
            raise ValueError("target_params must be a tuple of length 4 (alpha, eps, mask, clip).")
        if not isinstance(lmks, list) or not all(isinstance(lmk, str) for lmk in lmks):
            raise TypeError("lmk must be a list of strings.")
        if transformations is not None and not callable(transformations):
            raise TypeError("transformations must be callable or None.")
        
        self.dataframe = dataframe
        self.root = root
        self.alpha, self.eps, self.mask, self.clip = target_params
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
        image_path = os.path.join(self.root, self.dataframe.loc[idx, 'mscan'])
        volume, header = extract_image(image_path)
        volume = torch.from_numpy(volume)
        image_tensor = volume.unsqueeze(0)  # Add channel dimension (C, H, W, D)

        # Load the landmark file containing coordinates
        landmark_path = os.path.join(self.root, self.dataframe.loc[idx, 'mcsv'])
        landmark_df = pd.read_csv(landmark_path)

        target = torch.zeros((len(self.lmks), 2, *volume.shape), dtype=torch.float32)  # Initialize target tensor
        coord_tensor = torch.zeros((len(self.lmks), 3), dtype=torch.float32)  # Initialize coordinates tensor
        try:
            spacings = header.get('spacings')[:3]
        except:
            spacings = np.array([header['space directions'][0, 0], 
                          header['space directions'][1, 1], 
                          header['space directions'][2, 2]])
            
        # Extract the coordinates for the specified landmark
        for i, lmk in enumerate(self.lmks):
            if lmk not in landmark_df['label'].values:
                raise ValueError(f"Landmark '{lmk}' not found in {landmark_path}")

            # Surprise the model!
            # if lmk in ['enL']:
            #     landmark_row = landmark_df[landmark_df['label'] == 'enR']
            # elif lmk in ['enR']:
            #     landmark_row = landmark_df[landmark_df['label'] == 'enL']       
            # else: 
            landmark_row = landmark_df[landmark_df['label'] == lmk]
           
            if landmark_row.empty:
                raise ValueError(f"Landmark '{lmk}' not found in {landmark_path}")

            # Convert landmark coordinates to a tensor and adjust for pixel spacing
            coord = landmark_row[['x', 'y', 'z']].iloc[0].tolist()
            coord = coord / spacings  # Normalize by pixel spacings
            coord = torch.tensor(coord, dtype=torch.float32)
            coord_tensor[i] = coord  # Store the coordinates in the tensor
            # Coords are in voxel coordinates, now. 
            heatmap, distance = gaussian_heatmap.create_gaussian_heatmap(
                        coord, volume, alpha=self.alpha, eps=self.eps, clip=self.clip, mask=self.mask
                )  # Add dimension e.g, L, T, D, H, W
            
            target[i][0], target[i][1] = heatmap, distance  # Assign the generated target to the corresponding landmark index
            # plot_heatmaps_slices_from_coord([target[i][0], target[i][1], 1-target[i][0]], coord.numpy())
            # print(target.shape)
        # Ensure transformations are provided
        if self.transformations is None:
            raise ValueError("Transformations are required but not provided.")

        # Apply transformations and create a TorchIO Subject
        subject = tio.Subject(
            image=self.transformations(tio.ScalarImage(tensor=image_tensor)),  # Apply transformations to the image
            target=target,  # Target heatmap and/or distance map
            name=self.dataframe.loc[idx, 'nsid'],  # Metadata: full ID of the subject
            pid=self.dataframe.loc[idx, 'npid'],  # Metadata: patient ID
            spacings=spacings,  # Pixel spacings
            coords = coord_tensor,  # Coordinates of landmarks
            lmk=self.lmks,  # List of landmarks
            visibles=self.dataframe.loc[idx, 'visibles']  # Visibility flag list for landmarks.
        )

        return subject
