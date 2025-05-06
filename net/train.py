from tqdm import tqdm
import wandb

def train_one_ep(model, loader, criterion, optimizer, device, wandb_steps, use_wandb = False):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model being trained.
        loader (DataLoader): DataLoader for training data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (str): Device ('cpu' or 'cuda').
        wandb_steps (dict): Global wandb loss indices.

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
    t = tqdm(loader, desc='Initializing.............', total=len(loader))

    # Iterate over the DataLoader
    for ind, batch in enumerate(t):
        # Move images and targets to the specified device
        images = batch['image']['data'].to(device)
        targets = batch['target']['data'].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass: compute model predictions
        outputs = model(images)

        # Compute the loss
        if ind % 50 == 0:
            try:
                # Visualize the target distance matrix
                loss = criterion(outputs, targets, flag_visualize=False)
            except Exception as e:
                print(f"Error during visualization: {e}")
                print(f"Visualization is not implemented for this loss fn!")
        else:
            loss = criterion(outputs, targets)
        # Compute the mean loss
        # Backward pass: compute gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Update running loss and calculate average loss
        running_loss += loss.item()
        avg_loss = running_loss / (ind + 1)

        # Update the progress bar description with the running average loss
        t.set_description(desc='Running Average Loss: {:.4f}'.format(avg_loss))

        if use_wandb:
            # Log step loss and mean loss to Weights & Biases (wandb)
            wandb.log({'train/step_loss': loss.item(), 'train/step': wandb_steps['train_loss']})
            wandb.log({'train/mean_loss': avg_loss, 'train/step': wandb_steps['train_loss']})

            # Increment the wandb step counter
            wandb_steps['train_loss'] += 1

    # Return the average loss and updated wandb_steps
    return avg_loss, wandb_steps
