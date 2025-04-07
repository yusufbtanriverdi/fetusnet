from tqdm import tqdm
import wandb

# 1. Min-Max Normalization (Scales data between 0 and 1)
def minmax_normalize(data):
    import numpy as np
    """
    Normalize data to range [0, 1]
    Formula: X_norm = (X - min) / (max - min)
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))

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

        # from net.loss.utils import soft_joint_histogram
        # import matplotlib.pyplot as plt

        # hig = soft_joint_histogram(outputs, targets, bins=128).detach().cpu().numpy()
        # hgg = soft_joint_histogram(targets, targets, bins=128).detach().cpu().numpy()
        # plt.figure()
        # plt.subplot(121)
        # plt.imshow(minmax_normalize(hig) * 255)            
        # plt.subplot(122)
        # plt.imshow(minmax_normalize(hgg) * 255)
        # plt.show()
        
        running_loss += loss.item()
        avg_loss = running_loss / (ind + 1)
        t.set_description(desc='Running Average Loss: {:.4f}'.format(avg_loss))

        wandb.log({'train/step_loss': loss, 'train/step': wandb_steps['train_loss']})
        wandb.log({'train/mean_loss': avg_loss, 'train/step': wandb_steps['train_loss']})
        wandb_steps['train_loss'] += ind 

    return avg_loss, wandb_steps
