from tqdm import tqdm
import torch
import wandb
import nrrd
import os
from torch.nn.functional import sigmoid
import pandas as pd

from net.metrics.metrics_eval import compute_metrics, compute_heatmap_metrics, compute_landmark_metrics
from net.dataset.MyDataset import extract_image
from net.plot.average_expected_local_accuracy import average_expected_local_accuracy
from net.postprocess.utility.save_fscv_csv import save_fscv_csv
from net.postprocess.utility.where_is_landmark import get_peak_location


def infer_one_ep(model, loader, criterion, device, wandb_steps, eval=False, save_dir=None, use_wandb=False, radii_eval=40, radii_num=100, loss_eval = False):
    """
    Perform inference for one epoch.

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
    ep_outputs, ep_targets, ep_spacings = [], [], []

    # Progress bar for validation
    t = tqdm(loader, desc="Validating...", total=len(loader))

    # Disable gradient computation for validation
    with torch.no_grad():
        for ind, batch in enumerate(t):
            # Move input data and targets to the specified device
            images = batch['image']['data'].to(device)
            targets = batch['target']['data'].to(device)
            spacings = batch['spacings']

            # Forward pass through the model
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # Store outputs, targets, and spacings for later evaluation
            ep_outputs.append(outputs.cpu())
            ep_targets.append(targets.cpu())
            ep_spacings.append(spacings)

            # Compute running average loss
            avg_loss = running_loss / (ind + 1)
            t.set_description(f"Running Average Loss: {avg_loss:.4f}")

            if use_wandb:
                # Log loss to Weights & Biases
                wandb.log({'val/step_loss': loss.item(), 'val/step': wandb_steps['val_loss']})
                wandb.log({'val/mean_loss': avg_loss, 'val/step': wandb_steps['val_loss']})
                wandb_steps['val_loss'] += 1  # Increment step

    # Concatenate all stored tensors for evaluation
    ep_outputs = sigmoid(torch.cat(ep_outputs, dim=0).view(-1, 128, 128, 128))  # Reshape to match expected dimensions
    ep_targets = torch.cat(ep_targets, dim=0).view(-1, 128, 128, 128)
    # Compute evaluation metrics if required
    scores = compute_metrics(ep_outputs, ep_targets, ep_spacings)

    if use_wandb:
        for k, v in scores.items():
            wandb.log({f'epoc/{k}': v, 'epoc/epoch': wandb_steps['epoch']})
        wandb_steps['epoch'] += 1  # Increment epoch step
        # If not using wandb, set scores to None    
    else:
        print("Scores: ", scores)
    # If eval=True, save outputs and generate visualizations
    if eval:
        scores['aela'] = average_expected_local_accuracy(ep_outputs, ep_targets, ep_spacings, torch.linspace(0, radii_eval, radii_num), save_dir=os.path.join(save_dir, f"aela.png"))
        # Load template header for saving NRRD files
        template_header = extract_image('templates/1.nrrd')[1]

        for ind, batch in tqdm(enumerate(loader), desc="Saving outputs", total=len(loader)):
            if len(batch) != 1:
                continue
            
            output = ep_outputs[ind]
            target = ep_targets[ind]
            input = batch['image']['data'][0, 0]
            output[input == 0] = 0

            try:
                # Visualize the target distance matrix
                loss = criterion(outputs, targets, flag_visualize=loss_eval)
            except Exception as e:
                print(f"Error during visualization: {e}")
                print(f"Visualization is not implemented for this loss fn!")
                raise e
                # Visualize the target distance matrix
                loss = criterion(outputs, targets, flag_visualize=False)


    # Return the average loss, computed metrics, and updated wandb_steps
    return avg_loss, scores, wandb_steps

def infer_one_ep_v2(model, loader, criterion, device, wandb_steps, eval=False, save_dir=None, use_wandb=False, radius_eval=40, radius_num=100, extract_via='argmax', lmk='unknown'):
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
    ep_scores_curve = []
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
            # Get the voxel coordinates of the target landmark
            target_coord_x = batch['coord_x'][0].to(device)
            target_coord_y = batch['coord_y'][0].to(device)
            target_coord_z = batch['coord_z'][0].to(device)   

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

            output_heatmap = sigmoid(output_heatmap[0][0])  # Apply sigmoid to the output heatmap
            target_heatmap = target_heatmap[0][0]  
            output_coord_x, output_coord_y, output_coord_z = get_peak_location(output_heatmap, extract_via)  # Ensure peak locations are computed
            
            target_coord_tensor = torch.tensor([target_coord_x, target_coord_y, target_coord_z], device=device)
            output_coord_tensor = torch.tensor([output_coord_x, output_coord_y, output_coord_z], device=device)

            # Print the quantization error.
            # print(get_peak_location(target_heatmap, 'argmax'))
            # print(target_coord_tensor)
            # print(output_coord_tensor)

            scores_v1 = compute_heatmap_metrics(output_heatmap, target_heatmap)
            scores_v2 = compute_landmark_metrics(output_coord_tensor,
                                                 target_coord_tensor, 
                                                 spacing)

            scores = {**scores_v1, **scores_v2}
            ep_scores.append(scores)

            scores_v3_distances = average_expected_local_accuracy(output_heatmap, target_coord_tensor, spacing=spacing, 
                                                                  radius_eval=radius_eval, radius_num=radius_num, 
                                                                  save_dir=os.path.join(save_dir, f"curve.png"),
                                                                  extract_via=extract_via)
            
            # print(scores_v3_distances)
            ep_scores_curve.append(scores_v3_distances)

            if eval:
                # print(f"Scores for batch {ind}: {scores}")
                # Save outputs and generate visualizations
                template_header = extract_image('templates/1.nrrd')[1]
                name = batch['name'][0]  # Extract the name of the current sample
                # Create a subdirectory within save_dir for saving outputs
                output_dir = os.path.join(save_dir, "predictions")
                os.makedirs(output_dir, exist_ok=True)
                # Save the predicted output as an NRRD file in the subdirectory
                nrrd.write(
                    os.path.join(output_dir, f"{name}.nrrd"),
                    output_heatmap.cpu().numpy(),
                    header=template_header
                )
                save_fscv_csv(
                    out=os.path.join(output_dir, f"{name}"),
                    coords=output_coord_tensor.cpu().numpy(),
                    selected_lmk=lmk,  # Replace with actual landmark name if available
                    spacing=spacing
                )

                # Save target heatmap and coordinates
                template_header = extract_image('templates/1.nrrd')[1]
                name = batch['name'][0]  # Extract the name of the current sample
                # Create a subdirectory within save_dir for saving outputs
                output_dir = os.path.join(save_dir, "generations")
                os.makedirs(output_dir, exist_ok=True)
                # Save the predicted output as an NRRD file in the subdirectory
                nrrd.write(
                    os.path.join(output_dir, f"{name}.nrrd"),
                    target_heatmap.cpu().numpy(),
                    header=template_header
                )
                save_fscv_csv(
                    out=os.path.join(output_dir, f"{name}"),
                    coords=target_coord_tensor.cpu().numpy(),
                    selected_lmk=f'{lmk}',  # Replace with actual landmark name if available
                    spacing=spacing
                )


    # Return the average loss, computed metrics, and updated wandb_steps
    avg_loss = running_loss / len(loader)        

    # Stack the list of tensors before taking the mean to avoid ValueError
    ep_scores_curve = torch.stack(ep_scores_curve, dim=0).mean(dim=0)
    ep_scores = pd.DataFrame.from_records(ep_scores)

    print(ep_scores)
    return avg_loss, ep_scores, ep_scores_curve, wandb_steps