import torch.optim as optim
import torch
from functools import partial

from net.model.backbone import ResUNet3D  
from net.loss.voxel import *

def get_fresh_model(params):
    """
    Initialize a fresh model, loss function, and optimizer based on the provided parameters.

    This function sets up:
      - A 3D ResUNet model for medical image processing or landmark detection tasks.
      - A loss function chosen from several custom or standard options.
      - An optimizer configured with the desired learning rate and settings.

    Args:
        params (Namespace or dict-like): Configuration object with attributes:
            - lmks (list): List of target landmarks, used to set output channels.
            - num_fts (int, optional): Number of base features in the UNet (default: 64).
            - device (str, optional): Device for model training ('cuda' or 'cpu').
            - reduction (str, optional): Reduction type for the loss (e.g., 'mean', 'sum').
            - loss (list[str], optional): List with one string indicating the loss function name.
            - optimizer (str, optional): Optimizer name ('adam' or 'sgd').
            - learning_rate (float, optional): Learning rate for the optimizer (default: 1e-3).

    Returns:
        tuple:
            - model (torch.nn.Module): Initialized ResUNet3D model.
            - criterion (nn.Module): Selected loss function instance.
            - optimizer (torch.optim.Optimizer): Configured optimizer.
            - float: Placeholder value (currently set to `torch.inf`).
    """
    # === Model Initialization ===
    model = ResUNet3D.ResUNet3D(
        input_channels=1,  # Assuming single-channel input (e.g., grayscale or single-modality volumes)
        output_channels=len(params.lmks),  # Output channels match the number of landmarks
        base_features=getattr(params, "num_fts", 64)  # Base feature count (default: 64)
    )

    # Move model to the specified device (default to CUDA if available)
    device = getattr(params, "device", "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # === Loss Function Selection ===
    reduction = getattr(params, "reduction", 'mean')
    loss_dict = {
        'kld': KullbackLeiblerDivLoss,
        'mse': MeanSquaredErrorLoss,
        'crossentropy': SoftmaxCrossEntropyLoss,
        'distmatrix': DistanceMatrixLoss,
        'emd': EMDRegularizedLoss,
    }

    loss_name = getattr(params, 'loss', ['mse'])  # Default to 'mse' if not specified
    loss_key = loss_name[0]

    if loss_key not in loss_dict:
        raise ValueError(f"Unsupported loss function '{loss_key}'. Choose from: {list(loss_dict.keys())}.")

    # Dynamically instantiate the loss function with parameters
    loss_cls = loss_dict[loss_key]
    loss_params = getattr(params, 'loss_params', {})  # Additional parameters for the loss function
    criterion = loss_cls(reduction=reduction, **loss_params)
    criterion = loss_dict[loss_key]()  # Instantiate the selected loss function

    # === Optimizer Selection ===
    m = getattr(params, 'lr_momentum', 0.9)

    optim_dict = {
        'adam': optim.Adam,
        'sgd': partial(optim.SGD, momentum=m)
    }

    optim_name = getattr(params, 'optimizer', 'adam')
    optimizer_cls = optim_dict.get(optim_name)
    if optimizer_cls is None:
        raise ValueError(f"Unsupported optimizer '{optim_name}'. Choose from: {list(optim_dict.keys())}.")

    learning_rate = getattr(params, 'learning_rate', 1e-3)
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer, torch.inf
