from net.dataset.target.gaussian_heatmap import create_gaussian_heatmap
from net.dataset.MyDataset import extract_image
import os, torch
import pandas as pd
import nrrd
import time

def main(dataframe, exp_dir, params):
    for lmk in params.lmks:
        for i in params.target_idx:
            if params.generate == 'gaussian':
                start_time = time.time()  # Start timing
                # Load 3D volume
                image_path = os.path.join(params.sys + params.root, dataframe.loc[i, 'processed__vol_path'])
                volume, header = extract_image(image_path)
                # image_tensor = torch.tensor(volume).unsqueeze(0)  # Add channel dimension

                # Load landmark file
                landmark_path = os.path.join(params.sys +  params.root, dataframe.loc[i, 'processed__csv_path'])
                landmark_df = pd.read_csv(landmark_path)

                # Extract coordinates for the selected landmark
                landmark_row = landmark_df[landmark_df['label'] == lmk]
                if landmark_row.empty:
                    raise ValueError(f"Landmark '{lmk}' not found in {landmark_path}")

                coord = landmark_row[['x', 'y', 'z']].iloc[0].tolist()  # Convert to list
                coord = coord / header['spacings'][:3]
                coord_tensor = torch.abs(torch.tensor(coord, dtype=torch.float32)) 

                # Generate target heatmap

                target = create_gaussian_heatmap(coord_tensor, volume, alpha=params.alpha, eps=params.eps)
                end_time = time.time()  # End timing
                elapsed_time = end_time - start_time  # Compute elapsed time
                print(f"Processed {os.path.join(params.sys, params.root, params.dataset[0], dataframe.loc[i, 'processed__vol_path'])} in {elapsed_time:.4f} seconds")
                # Save target
                nrrd.write(os.path.join(exp_dir, f"{dataframe.loc[i, 'full_id']}.nrrd"), target, header=header)
                
