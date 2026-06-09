# Import necessary modules
from net.dataset.target.gaussian_heatmap import create_gaussian_heatmap
from net.dataset.MyDataset import extract_image
import os
import torch
import pandas as pd
import nrrd
import time
import numpy as np

def perform_generate(sinfo, exp_dir, params):
    """
    Main function to generate target heatmaps or distance matrices for 3D volumes.

    Args:
        sinfo (pd.DataFrame): DataFrame containing information about the dataset.
        exp_dir (str): Directory to save the generated targets.
        params (Namespace): Parameters including landmarks, test patients, and generation settings.
    """
    for lmk in params.target_.lmks:  # Iterate over the list of landmarks
        # Determine the indices of the target patients
        if params.test_patients:
            target_idx = sinfo.index[sinfo['pid'].isin(params.test_patients)].tolist()
            print(target_idx)
        else:
            raise ValueError("No test patients specified in params.test_patients")
        
        for i in target_idx:  # Iterate over the target indices
            start_time = time.time()  # Start timing

            # Load the 3D volume
            image_path = os.path.join(params.dataset_.sys + params.dataset_.root, sinfo.loc[i, 'mscan'])
            volume, header = extract_image(image_path)

            # Load the landmark file
            landmark_path = os.path.join(params.dataset_.sys + params.dataset_.root, sinfo.loc[i, 'mcsv'])
            landmark_df = pd.read_csv(landmark_path)

            # Extract coordinates for the selected landmark
            landmark_row = landmark_df[landmark_df['label'] == lmk]
            if landmark_row.empty:
                raise ValueError(f"Landmark '{lmk}' not found in {landmark_path}")

            try:
                spacings = header.get('spacings')[:3]
            except:
                spacings = np.array([header['space directions'][0, 0], 
                            header['space directions'][1, 1], 
                            header['space directions'][2, 2]])
            # Convert coordinates to a tensor and adjust for spacing
            coord = landmark_row[['x', 'y', 'z']].iloc[0].tolist()  # Convert to list
            coord = coord / spacings # Adjust for voxel spacing
            coord_tensor = torch.abs(torch.tensor(coord, dtype=torch.float32))

            # Generate the target (heatmap or distance matrix)
            target, distance = create_gaussian_heatmap(coord_tensor, volume, alpha=params.g_alpha, eps=params.g_eps, clip=params.clip, mask=params.mask)
 
            end_time = time.time()  # End timing
            elapsed_time = end_time - start_time  # Compute elapsed time
            print(f"Processed {sinfo.loc[i, 'nsid']} in {elapsed_time:.4f} seconds")

            # Save the generated target to a file
            nrrd.write(os.path.join(exp_dir, f"{sinfo.loc[i, 'nsid']}_{lmk}.nrrd"), target, header=header)
