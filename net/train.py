from tqdm import tqdm
import wandb


def train_one_ep(model, loader, criterion, optimizer, device, wandb_steps):
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
        dict: Training metrics (e.g., accuracy).
    """
    model.train()
    running_loss = 0.0
    # all_metrics = []

    avg_loss = -1
    
    t = tqdm(loader, desc='Initializing.............', total=len(loader))
    for ind, batch in enumerate(t):

        images, targets = batch['image']['data'].to(device), batch['target']['data'].to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        avg_loss = running_loss / (ind + 1)
        t.set_description(desc='Running Average Loss: {:.2f}'.format(avg_loss))

        wandb.log({'train/step_loss': loss, 'train/step': wandb_steps['train_loss']})
        wandb.log({'train/mean_loss': avg_loss, 'train/step': wandb_steps['train_loss']})
        wandb_steps['train_loss'] += ind 

    return avg_loss, wandb_steps
