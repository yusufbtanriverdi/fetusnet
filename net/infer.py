from tqdm import tqdm
import torch
import wandb
import nrrd, os

from net.metrics.metrics_eval import compute_metrics
from net.dataset.MyDataset import extract_image
from net.visual.detection.planes import overlay_heatmaps

def infer_one_ep(model, loader, criterion, device, wandb_steps, eval=False, save_dir=None):
    """
    Validate the model for one epoch.
    
    Args:
        model (torch.nn.Module): The model being validated.
        loader (DataLoader): DataLoader for validation data.
        criterion (torch.nn.Module): Loss function.
        device (str): Device ('cpu' or 'cuda').
        wandb_steps (dict): Global wandb loss indices.
        eval (bool, optional): Whether to compute evaluation metrics. Default is False.
        
    Returns:
        float: Average validation loss for the epoch.
        dict: Validation metrics (if eval=True).
    """

    model.eval()
    running_loss = 0.0
    ep_outputs, ep_targets, ep_spacings = [], [], []

    t = tqdm(loader, desc="Validating...", total=len(loader))

    with torch.no_grad():
        for ind, batch in enumerate(t):
            images = batch['image']['data'].to(device)
            targets = batch['target']['data'].to(device)
            spacings = batch['spacings']

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # Store outputs and targets for later evaluation
            ep_outputs.append(outputs.cpu())  
            ep_targets.append(targets.cpu())
            ep_spacings.append(spacings)

            # Compute running average loss
            avg_loss = running_loss / (ind + 1)
            t.set_description(f"Running Average Loss: {avg_loss:.4f}")

            # Log loss to Weights & Biases
            wandb.log({'val/step_loss': loss.item(), 'val/step': wandb_steps['val_loss']})
            wandb.log({'val/mean_loss': avg_loss, 'val/step': wandb_steps['val_loss']})
            wandb_steps['val_loss'] += 1  # Increment step

    # Concatenate all stored tensors
    ep_outputs = torch.cat(ep_outputs, dim=0).view(-1, 128, 128, 128) # number of batches * batch_size, 128, 128, 128
    ep_targets = torch.cat(ep_targets, dim=0).view(-1, 128, 128, 128)

    # Compute evaluation metrics if required
    scores = {}
    scores = compute_metrics(ep_outputs, ep_targets, ep_spacings)
    for k, v in scores.items():
        wandb.log({f'epoc/{k}': v, 'epoc/epoch': wandb_steps['epoch']})
    wandb_steps['epoch'] += 1  # Increment step
    
    if eval:
        template_header = extract_image('templates/template.nrrd')[1]
        for ind, batch in enumerate(loader):
            name = batch['name'][0]
            nrrd.write(os.path.join(save_dir, f"{name}.nrrd"), ep_outputs[ind].cpu().numpy(), header=template_header)
            fig = overlay_heatmaps(batch['image']['data'][0].cpu().numpy(),
                                   batch['target']['data'][0].cpu().numpy(),
                                   ep_outputs[ind].cpu().numpy() # assuming batch size = 1
                                   )
            os.makedirs(os.path.join(save_dir, 'detection_planes'), exist_ok=True)
            fig.savefig(os.path.join(save_dir, 'detection_planes', f"{name}.nrrd"))
            
    return avg_loss, scores, wandb_steps
