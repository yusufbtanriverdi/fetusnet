from tqdm import tqdm
import torch
import wandb
import nrrd
import os
from torch.nn.functional import sigmoid

from net.metrics.metrics_eval import compute_metrics
from net.dataset.MyDataset import extract_image
from net.visual.detection.planes import overlay_heatmaps

def infer_one_ep(model, loader, criterion, device, wandb_steps, eval=False, save_dir=None, use_wandb=False):
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
    ep_outputs = torch.cat(ep_outputs, dim=0).view(-1, 128, 128, 128)  # Reshape to match expected dimensions
    ep_targets = torch.cat(ep_targets, dim=0).view(-1, 128, 128, 128)

    if use_wandb:
        # Compute evaluation metrics if required
        scores = compute_metrics(ep_outputs, ep_targets, ep_spacings)
        for k, v in scores.items():
            wandb.log({f'epoc/{k}': v, 'epoc/epoch': wandb_steps['epoch']})
        wandb_steps['epoch'] += 1  # Increment epoch step
    else:  
        scores = None
        # If not using wandb, set scores to None    

    # If eval=True, save outputs and generate visualizations
    if eval:
        # Load template header for saving NRRD files
        template_header = extract_image('templates/template.nrrd')[1]

        for ind, batch in tqdm(enumerate(loader), desc="Saving outputs", total=len(loader)):
            name = batch['name'][0]  # Extract the name of the current sample

            # Save the predicted output as an NRRD file
            nrrd.write(
                os.path.join(save_dir, f"{name}.nrrd"),
                sigmoid(ep_outputs[ind]).cpu().numpy(),
                header=template_header
            )

            # Generate and save detection plane visualizations
            fig = overlay_heatmaps(
                batch['image']['data'][0, 0].cpu().numpy(),
                batch['target']['data'][0, 0].cpu().numpy(),
                sigmoid(ep_outputs[ind].cpu()).cpu().numpy()  # Assuming batch size = 1
            )
            os.makedirs(os.path.join(save_dir, 'detection_planes'), exist_ok=True)
            fig.savefig(os.path.join(save_dir, 'detection_planes', f"{name}.png"))
    # Return the average loss, computed metrics, and updated wandb_steps
    return avg_loss, scores, wandb_steps
