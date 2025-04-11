# Import necessary modules
from net.dataset.target.gaussian_heatmap import create_gaussian_heatmap
from net.dataset.target.distance_matrix import create_distance_matrix
from net.dataset.MyDataset import extract_image
import os
import torch
import pandas as pd
import nrrd
import time

def main(sinfo, exp_dir, params):
    """
    Main function to generate target heatmaps or distance matrices for 3D volumes.

    Args:
        sinfo (pd.DataFrame): DataFrame containing information about the dataset.
        exp_dir (str): Directory to save the generated targets.
        params (Namespace): Parameters including landmarks, test patients, and generation settings.
    """
    for lmk in params.lmks:  # Iterate over the list of landmarks
        # Determine the indices of the target patients
        if params.test_patients:
            target_idx = sinfo.index[sinfo['pid'].isin(params.test_patients)].tolist()
        else:
            target_idx = params.target_idx

        for i in target_idx:  # Iterate over the target indices
            start_time = time.time()  # Start timing

            # Load the 3D volume
            image_path = os.path.join(params.sys + params.root, sinfo.loc[i, 'processed__vol_path'])
            volume, header = extract_image(image_path)

            # Load the landmark file
            landmark_path = os.path.join(params.sys + params.root, sinfo.loc[i, 'processed__csv_path'])
            landmark_df = pd.read_csv(landmark_path)

            # Extract coordinates for the selected landmark
            landmark_row = landmark_df[landmark_df['label'] == lmk]
            if landmark_row.empty:
                raise ValueError(f"Landmark '{lmk}' not found in {landmark_path}")

            # Convert coordinates to a tensor and adjust for spacing
            coord = landmark_row[['x', 'y', 'z']].iloc[0].tolist()  # Convert to list
            coord = coord / header['spacings'][:3]  # Adjust for voxel spacing
            coord_tensor = torch.abs(torch.tensor(coord, dtype=torch.float32))

            # Generate the target (heatmap or distance matrix)
            if params.generate == 'gaussian':
                target = create_gaussian_heatmap(coord_tensor, volume, alpha=params.alpha, eps=params.eps)
            else:
                target = create_distance_matrix(coord_tensor, volume, alpha=params.alpha, eps=params.eps)

            end_time = time.time()  # End timing
            elapsed_time = end_time - start_time  # Compute elapsed time
            print(f"Processed {os.path.join(params.sys, params.root, params.dataset[0], sinfo.loc[i, 'processed__vol_path'])} in {elapsed_time:.4f} seconds")

            # Save the generated target to a file
            nrrd.write(os.path.join(exp_dir, f"{sinfo.loc[i, 'full_id']}.nrrd"), target, header=header)
