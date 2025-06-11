from tqdm import tqdm
import torch
import wandb
import nrrd
import os
from torch.nn.functional import sigmoid
import pandas as pd
import numpy as np

from net.metrics.metrics_eval import compute_heatmap_metrics, compute_landmark_metrics
from net.dataset.MyDataset import extract_image
from net.plot.average_expected_local_accuracy import average_expected_local_accuracy, plot_aela_figure
from net.postprocess.utility.save_fscv_csv import save_fscv_csv
from net.postprocess.utility.where_is_landmark import get_peak_location
from net.plot.histogram_flattened import plot_histograms_and_stats

def infer_one_ep(model, loader, criterion, device, wandb_steps, use_wandb=False, extract_via='argmax', eval=False, save_dir=None, radius_eval=40, radius_num=100, lmks=None):
    """
    Perform inference for one epoch with additional functionality.

    Args:
        model (torch.nn.Module): The model to validate.
        loader (DataLoader): DataLoader for validation data.
        criterion (torch.nn.Module): Loss function.
        device (str): Device to run the model on ('cpu' or 'cuda').
        wandb_steps (dict): Dictionary to track Weights & Biases (wandb) steps.
        eval (bool, optional): Whether to compute evaluation metrics and save outputs. Default is False.
        save_dir (str, optional): Directory to save outputs if eval=True. Default is None.

    Returns:
        float: Average validation loss for the epoch.
        dict: Validation metrics (if eval=True).
        dict: Updated wandb_steps.
    """
    
    # Set the model to evaluation mode
    model.eval()

    # Initialize variables to track loss and outputs
    running_loss = 0.0

    # Progress bar for validation
    t = tqdm(loader, desc="Validating...", total=len(loader))
    ep_scores = []
    distance_curves = []
    # Disable gradient computation for validation
    with torch.no_grad():
        print("Starting validation loop...")
        for ind, batch in enumerate(t): # Assuming batch_size = 1 for simplicity
            if len(batch['image']['data']) != 1:
                print(f"Skipping batch {ind} due to unexpected batch size: {len(batch)}")
                continue
            # Move input data and targets to the specified device
            image = batch['image']['data'].to(device)
            target_heatmap = batch['target']['data'].to(device)
            spacing = batch['spacings'][0][0] # Assuming ISO spacing for simplicity

            # target_coord_x = batch['coord_x'][0].to(device)
            # target_coord_y = batch['coord_y'][0].to(device)
            # target_coord_z = batch['coord_z'][0].to(device)   

            # assert (target_coord_x, target_coord_x, target_coord_z ) == get_peak_location(target_heatmap, 'argmax')  # Ensure peak locations are computed
            # Forward pass through the model
            output_heatmap = model(image)

            # Compute loss
            loss = criterion(output_heatmap, target_heatmap)
            running_loss += loss.item()
            # Compute running average loss
            avg_loss = running_loss / (ind + 1)
            t.set_description(f"Running Average Loss: {avg_loss:.4f}")
            if use_wandb:
                # Log loss to Weights & Biases
                wandb.log({'val/step_loss': loss.item(), 'val/step': wandb_steps['val_loss']})
                wandb.log({'val/mean_loss': avg_loss, 'val/step': wandb_steps['val_loss']})
                wandb_steps['val_loss'] += 1  # Increment step
            # Print the quantization error.
            # print(get_peak_location(target_heatmap, 'argmax'))
            # print(target_coord_tensor)
            # print(output_coord_tensor)

            output_heatmap = sigmoid(output_heatmap[0])  # Apply sigmoid to the output heatmap from batch size 1
            target_heatmap = target_heatmap[0]  # Extract the target heatmap from batch size 1
            scores_v1 = compute_heatmap_metrics(output_heatmap, target_heatmap)

            # Compute the peak locations of the output heatmap
            output_coord_tensor = get_peak_location(output_heatmap, extract_via).to(device)  # Ensure peak locations are computed
            # Get the voxel coordinates of the target landmark
            target_coord_tensor = batch['coords'][0].to(device)  # Assuming batch size of 1

            scores_v2 = compute_landmark_metrics(output_coord_tensor,
                                                 target_coord_tensor, 
                                                 spacing)

            scores = {**scores_v1, **scores_v2}
            name = batch['name'][0]  # Extract the name of the current sample from batch size 1
            scores['fname'] = name  # Add the name to the scores dictionary
            ep_scores.append(scores)
            if eval:
                output_dir = os.path.join(save_dir, "predictions")
                os.makedirs(output_dir, exist_ok=True)
                save_fscv_csv(
                        out=os.path.join(output_dir, f"{name}"),
                        coords=output_coord_tensor.cpu().numpy(),
                        selected_lmks=lmks,  # Replace with actual landmark name if available
                        spacing=spacing
                    )
                
                target_dir = os.path.join(save_dir, "generations")
                os.makedirs(target_dir, exist_ok=True)
                save_fscv_csv(
                        out=os.path.join(target_dir, f"{name}"),
                        coords=output_coord_tensor.cpu().numpy(),
                        selected_lmks=lmks,  # Replace with actual landmark name if available
                        spacing=spacing
                    )                
                
                for i, lmk in enumerate(lmks):

                    # Extract the landmark coordinates for the current landmark
                    scores_v3_distances = average_expected_local_accuracy(output_heatmap[i].unsqueeze(0), target_coord_tensor[i].cpu(), 
                                                                          spacing=spacing, 
                                                                          radius_eval=radius_eval, radius_num=radius_num, 
                                                                          save_dir=None,
                                                                          extract_via=extract_via)
                    distance_curves.append(scores_v3_distances)
                    # Save outputs and generate visualizations
                    template_header = extract_image('templates/1.nrrd')[1]
                    # Save the predicted output as an NRRD file in the subdirectory
                    nrrd.write(
                        os.path.join(output_dir, f"{name}_{lmk}.nrrd"),
                        output_heatmap[i].cpu().numpy(),
                        header=template_header
                    )
                    # Save the predicted output as an NRRD file in the subdirectory
                    nrrd.write(
                        os.path.join(target_dir, f"{name}_{lmk}.nrrd"),
                        target_heatmap[i].cpu().numpy(),
                        header=template_header
                    )


                plot_histograms_and_stats(output_heatmap, target_heatmap, save_path = os.path.join(output_dir, f"{name}_"))
        
        if eval:
            curves_dir = os.path.join(save_dir, "curves")
            os.makedirs(curves_dir, exist_ok=True)
            # Save the ep_scores_curve as CSV, including lmk info in the filename
            distance_curves = torch.stack(distance_curves, dim=0).mean(dim=0).tolist()
            curve_csv_path = os.path.join(curves_dir, f"{lmk}_curve_mean.csv") 
            np.savetxt(curve_csv_path, distance_curves, delimiter=",")
            plot_aela_figure(torch.linspace(0, radius_eval, radius_num), distance_curves, save_dir=os.path.join(curves_dir, f"{lmk}_curve_mean.png") )
        
    # Return the average loss, computed metrics, and updated wandb_steps
    avg_loss = running_loss / len(loader)        
    
    ep_scores = pd.DataFrame.from_records(ep_scores)
    return avg_loss, ep_scores, wandb_steps