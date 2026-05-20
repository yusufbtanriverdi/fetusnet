from tqdm import tqdm
import wandb

def train_one_ep(model, loader, criteria, multi_loss, optimizer, device, wandb_steps, use_wandb, progress_bar):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model being trained.
        loader (torch.utils.data.DataLoader): DataLoader for training data.
        criteria (callable): Loss functions.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device or str): Device to run training on ('cpu' or 'cuda').
        wandb_steps (dict): Dictionary tracking wandb step indices.
        use_wandb (bool, optional): Whether to log metrics to Weights & Biases. Default is False.
        progress_bar (bool, optional): Whether to display a progress bar. Default is True.

    Returns:
        float: Average loss for the epoch.
        dict: Updated wandb_steps with training metrics.
    """
    # Set the model to training mode
    model.train()

    # Initialize running loss and average loss
    running_loss = 0.0
    avg_loss = -1

    # Create a progress bar for the DataLoader
    t = tqdm(loader, desc="Training...", total=len(loader)) if progress_bar else loader

    # Iterate over the DataLoader
    for ind, batch in enumerate(t):
        # Move images and targets to the specified device
        images = batch['image']['data'].to(device)
        targets = batch['target'].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass: compute model predictions
        outputs = model(images)
        # Compute the losses
        losses = [criterion(outputs, targets) for criterion in criteria]
        if multi_loss:
            loss = multi_loss(losses)
        else: 
            loss = sum(losses)
        # Compute the mean loss
        # Backward pass: compute gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Update running loss and calculate average loss
        running_loss += loss.item()
        avg_loss = running_loss / (ind + 1)

        # Update the progress bar description with the running average loss
        if progress_bar:
            t.set_description(desc='Running Average Loss: {:.4f}'.format(avg_loss)) 

        if use_wandb:
            # Log step loss and mean loss to Weights & Biases (wandb)
            wandb.log({'train/step_loss': loss.item(), 'train/step': wandb_steps['train_loss']})
            wandb.log({'train/mean_loss': avg_loss, 'train/step': wandb_steps['train_loss']})
            for i in range(len(losses)):
                wandb.log({f'train/noise_params_{i}': multi_loss.noise_params[i], 'train/step': wandb_steps['train_loss']})
            # Increment the wandb step counter
            wandb_steps['train_loss'] += 1

    return avg_loss, wandb_steps
