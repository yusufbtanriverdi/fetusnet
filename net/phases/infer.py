from tqdm import tqdm
import torch
import wandb
import os
import pandas as pd
import numpy as np
import gc 
import nrrd 

from net.metrics.metrics_eval import compute_heatmap_metrics, compute_landmark_metrics
from net.plot.curves import compute_aela, plot_aela_figure
from net.plot.histograms import plot_histograms_and_stats
from net.plot.matrices_3d import plot_3d_matrices
from net.plot.heatmaps import plot_heatmaps_slices_from_coord

from net.postprocess.utility.save_fscv_csv import save_fscv_csv
from net.postprocess.utility.where_is_landmark import get_peak_location
from net.dataset.utility.rotation import extract_image
from net.plot.heatmaps import plot_heatmaps_slices_from_coord # To debug.
from net.loss.losses import *

def infer_one_ep(model, loader, criteria, device, wandb_steps, use_wandb, detector, progress_bar,
                 lmks, experiment_dir, check_visibility, eval, **kwargs):
    """
    Perform inference for one epoch with additional functionality.

    Args:
        model (torch.nn.Module): The model to validate.
        loader (DataLoader): DataLoader for validation data.
        criteria (torch.nn.Module): Loss functions.
        device (str or torch.device): Device to run the model on ('cpu' or 'cuda').
        wandb_steps (dict): Dictionary to track Weights & Biases (wandb) steps.
        use_wandb (bool, optional): Whether to log metrics to Weights & Biases. Default is False.
        detector (str, optional): Method to detect peaks in heatmaps ('argmax', etc.). Default is 'argmax'.
        progress_bar (bool, optional): Whether to display a progress bar. Default is True.
        experiment_dir (str, optional): Directory to save outputs if eval=True. Default is None.         
        lmks (list, optional): List of landmark names. Default is None.
        eval (bool, optional): Whether to compute evaluation metrics and save outputs. Default is False.
        
    Returns:
        float: Average validation loss for the epoch.
        dict: Updated wandb_steps.
        pd.DataFrame: DataFrame containing evaluation scores if eval=True, else empty DataFrame.
    """

    # Set the model to evaluation mode
    model.eval()

    # Initialize variables to track loss and outputs
    running_loss = 0.0
    avg_loss = -1

    # Create a progress bar for the DataLoader
    t = tqdm(loader, desc="Validating...", total=len(loader)) if progress_bar else loader

    ep_scores = []  # List to store scores for each epoch    
    if eval: 
        radius_num = kwargs.get('radius_num', 100)
        distance_curves = torch.zeros((len(lmks), len(t), radius_num), dtype=torch.float32)  # Initialize distance curves tensor
    
    # Disable gradient computation for validation
    with torch.no_grad():
        # Iterate over the DataLoader
        for ind, batch in enumerate(t): 
            if len(batch['image']['data']) != 1:
                print(f"Skipping batch {ind} due to unexpected batch size: {len(batch)}")
                continue
            # Move input data and targets to the specified device
            images = batch['image']['data'].to(device)
            targets = batch['target'].to(device)
            spacing = batch['spacings'][0][0] # Assuming ISO spacing for simplicity
            nsid = batch['name'][0]
            visibles = batch['visibles'][0]  # Assuming batch size of 1
            # Forward pass through the model
            outputs = model(images)

            # Compute the losses
            losses = [criterion(outputs, targets) for criterion in criteria]
            loss = losses[0]

            # Update running loss and calculate average loss
            running_loss += loss.item()
            avg_loss = running_loss / (ind + 1)

            # output_heatmap = sigmoid(output_heatmap[0])             # Assuming batch size of 1   
            output_heatmap = outputs[0]            # Assuming batch size of 1   
            output_coord_tensor = get_peak_location(output_heatmap, detector).to(device)  # Ensure peak locations are computed
            target_coord_tensor = batch['coords'][0].to(device)   # Get the voxel coordinates of the target landmark, assuming batch size of 1            
            # Compute metrics for each landmark
            landmark_scores = {}
            for i, lmk in enumerate(lmks):
                landmark_score = compute_landmark_metrics(
                    output_coord_tensor[i],
                    target_coord_tensor[i], 
                    spacing
                )

                if check_visibility and lmk not in visibles:
                    landmark_score = {k: np.nan for k in landmark_score.keys()}

                landmark_scores.update({f"{k}_{lmk}": v for k, v in landmark_score.items()})

            dmean =np.nanmean(list(landmark_scores.values()))
            # Update the progress bar description with the running average loss
            if progress_bar:
                t.set_description(f"Running Average Loss: {avg_loss:.4f}")
            
            if use_wandb:
                # Log step loss and mean loss to Weights & Biases (wandb)
                wandb.log({'val/step_loss': loss.item(), 'val/step': wandb_steps['val_loss']})
                wandb.log({'val/mean_loss': avg_loss, 'val/step': wandb_steps['val_loss']})
                wandb.log({'val/dmean': dmean, 'val/step': wandb_steps['val_loss']})
                # Increment the wandb step counter
                wandb_steps['val_loss'] += 1 

            row = {'nsid': nsid, 'loss': loss.item(), 'dmean': dmean}
            row.update(landmark_scores)
            
            if eval:
                radius_eval = kwargs.get('radius_eval', 40)
                save_targets = kwargs.get('save_targets', False)
                save_outputs = kwargs.get('save_outputs', False)
                save_errormaps = kwargs.get('save_errormaps', False)
                show_figures = kwargs.get('show_figures', False)
                output_dir = os.path.join(experiment_dir, "eval")
                os.makedirs(output_dir, exist_ok=True)

                target_heatmap = targets[0] # {!} Assuming batch size of 1   
                for i, lmk in enumerate(lmks):
                    heatmap_scores = compute_heatmap_metrics(output_heatmap[i], 
                                                             target_heatmap[i]
                                                            )
                    
                    row.update({f"{lmk}_{k}": v for k, v in heatmap_scores.items()})

                save_fscv_csv(
                        out=os.path.join(output_dir, f"{nsid}"),
                        coords=output_coord_tensor.cpu().numpy(),
                        selected_lmks=lmks,  
                        spacing=spacing,
                    )

                for i, lmk in enumerate(lmks):
                    output_heatmap_i = torch.nn.functional.sigmoid(output_heatmap[i]) # {!} Assuming batch size of 1 

                    if check_visibility:
                        if lmk in visibles:
                            # Extract the landmark coordinates for the current landmark
                            scores_v3_distances = compute_aela(output_heatmap_i.unsqueeze(0), target_coord_tensor[i].cpu(), 
                                                                                spacing=spacing, 
                                                                                radius_eval=radius_eval, 
                                                                                radius_num=radius_num, 
                                                                                save_dir=None,
                                                                                detector='argmax',
                                                                                show=show_figures)
                        else:  
                            scores_v3_distances = torch.full((radius_num,), torch.nan)  # Placeholder value (currently set to `torch.nan`).
                    else:
                        # Extract the landmark coordinates for the current landmark
                        scores_v3_distances = compute_aela(output_heatmap_i.unsqueeze(0), target_coord_tensor[i].cpu(), 
                                                                            spacing=spacing, 
                                                                            radius_eval=radius_eval, 
                                                                            radius_num=radius_num, 
                                                                            save_dir=None,
                                                                            detector='argmax',
                                                                            show=show_figures)
                        
                    distance_curves[i, ind, :] = scores_v3_distances  # Store the distance curves for each landmark and batch index

                    if save_errormaps:
                        sse = SSELoss()
                        sce = SoftmaxCELoss()
                        emd = EucEMDLoss()
                        sse_error = sse.forward(outputs, targets).squeeze(0)
                        sce_error = sce.forward(outputs, targets).squeeze(0)
                        emd_error = emd.forward(outputs, targets).squeeze(0)

                    if show_figures:
                        _ = plot_histograms_and_stats(output_heatmap_i, target_heatmap[i])
                        plot_3d_matrices(output_heatmap_i, target_heatmap[i])
                        plot_heatmaps_slices_from_coord([
                            output_heatmap_i, 
                            target_heatmap[i],
                            sse_error[i],
                            sce_error[i],
                            emd_error[i]
                            ], 
                            coord_tensor=target_coord_tensor[i].cpu().numpy(), 
                            titles=[f"{lmk} Output", 
                                    f"{lmk} Target",
                                    f"{lmk} SSE Map",
                                    f"{lmk} Softmax CE Map",
                                    f"{lmk} Distance Penalty",
                                    ])

                    # Save the predicted output as an NRRD file in the subdirectory
                    if save_outputs:
                        template_header = extract_image('templates/1.nrrd')[1]
                        nrrd.write(
                            os.path.join(output_dir, f"{nsid}_{lmk}.nrrd"),
                            output_heatmap_i.cpu().numpy(),
                            header=template_header
                        )
                    
                    if save_targets:   
                        template_header = extract_image('templates/1.nrrd')[1]
                        nrrd.write(
                            os.path.join(output_dir, f"{nsid}_{lmk}_target.nrrd"),
                            target_heatmap[i].cpu().numpy(),
                            header=template_header
                        )
                        
                    gc.collect()
            ep_scores.append(row)
            # if ind > 100: break      # {!} For debugging, remove this line in production 
    if eval:
        for i, lmk in enumerate(lmks):
            # Save the ep_scores_curve as CSV, including lmk info in the filename
            curve_csv_path = os.path.join(output_dir, f"{lmk}_curve_mean.csv") 
            np.savetxt(curve_csv_path, distance_curves[i].nanmean(dim=0), delimiter=",")
            plot_aela_figure(torch.linspace(0, radius_eval, radius_num), distance_curves[i].mean(dim=0), save_dir=os.path.join(output_dir, f"{lmk}_curve_mean.png") )
            gc.collect()
        
        
    # Return the average loss, computed metrics, and updated wandb_steps
    avg_loss = running_loss / len(loader) 
    # Save ep_scores to a CSV file in the experiment directory and also save the mean scores across all samples.
    ep_scores = pd.DataFrame.from_records(ep_scores)
    ep_scores.to_csv(os.path.join(experiment_dir, "test_scores.csv"), index=False)
    ep_scores.drop(['nsid'], axis=1).mean().to_csv(os.path.join(experiment_dir, "test_scores_mean.csv"))

    return avg_loss, wandb_steps, ep_scores