from net.dataset.target.gaussian_heatmap import create_gaussian_heatmap
from net.dataset.MyDataset import extract_image
import os, torch
import pandas as pd
import nrrd

def main(dataframe, exp_dir, params):
    for lmk in params.lmks:
        for i in range(len(params.target_idx)):
            if params.target_mode == 'gaussian':
                # Load 3D volume
                image_path = os.path.join(params.root, dataframe.loc[i, 'heatmap_dts_vol_path'])
                volume, header = extract_image(image_path)
                image_tensor = torch.tensor(volume).unsqueeze(0)  # Add channel dimension

                # Load landmark file
                landmark_path = os.path.join(params.root, dataframe.loc[i, 'heatmap_dts_lmk_path'])
                landmark_df = pd.read_csv(landmark_path)

                # Extract coordinates for the selected landmark
                landmark_row = landmark_df[landmark_df['label'] == lmk]
                if landmark_row.empty:
                    raise ValueError(f"Landmark '{lmk}' not found in {landmark_path}")

                coord = landmark_row[['x', 'y', 'z']].iloc[0].tolist()  # Convert to list
                coord_tensor = torch.tensor(coord, dtype=torch.float32)

                # Generate target heatmap
                if params.target_mode == 'gaussian':
                    target = create_gaussian_heatmap(coord_tensor, volume, alpha=params.alpha, eps=params.eps)

                # Save target
                nrrd.write(os.path.join(exp_dir, f"{dataframe.loc[i, 'full_id']}.nrrd"), target, header=header)
                
