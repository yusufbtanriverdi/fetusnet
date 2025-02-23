import torch, os

def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_name=None):
    """
    Save a model checkpoint to the specified directory.

    Args:
        model (torch.nn.Module): The model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        epoch (int): The current epoch number.
        val_loss (float): The validation loss at the current epoch.
        val_recall (float): The validation recall observed during validation.
        checkpoint_dir (str): Directory path to save the checkpoint.
    """
    # Create checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }

    # Save checkpoint
    if not checkpoint_name:
        checkpoint_path = os.path.join(checkpoint_path, f'checkpoint_ep{epoch}.pt')
    else:
        checkpoint_path = checkpoint_name

    torch.save(checkpoint, checkpoint_path)
    # print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load a model checkpoint from the specified file path.

    Args:
        model (torch.nn.Module): The model to be loaded.
        optimizer (torch.optim.Optimizer): The optimizer to be loaded.
        checkpoint_path (str): File path of the checkpoint to load.

    Returns:
        tuple: Tuple containing the loaded model, optimizer, epoch number, and best validation loss.
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Load model state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state dictionary
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Additional information
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']

    return model, optimizer, epoch, val_loss



