import torch.optim as optim
import torch
from functools import partial

from net.model.backbone import ResUNet3D
from net.loss.losses import *
from utils import flatten_namespace_to_dict

def get_fresh_model(params):
    """
    Initialize a fresh model, loss function, and optimizer based on the provided parameters.

    Args:
        params (Namespace or dict-like): Configuration object with attributes.

    Returns:
        tuple:
            - model (torch.nn.Module): Initialized ResUNet3D model.
            - criterion (nn.Module): Selected loss function instance.
            - optimizer (torch.optim.Optimizer): Configured optimizer.
            - float: Placeholder value (currently set to `torch.inf`).
    """
    # === Model Initialization ===
    model_key = params.training.architecture  # Default to 'mse' if not specified.
    
    if model_key == 'resunet3d':
        model = ResUNet3D.ResUNet3D(
            input_channels=1,  # Assuming single-channel input (e.g., grayscale or single-modality volumes).
            output_channels=len(params.target.lmks),  # Output channels match the number of landmarks.
            base_features=params.training.num_fts,  # Base feature count (default: 32). 
        ) 
    else: 
        raise ValueError(f"Unsupported model architecture '{model_key}'. Choose from: ['resunet3d', 'resnet34'].")
    
    # Move model to the specified device (default to CUDA if available).
    device = params.device
    model.to(device)
    # === Loss Function Selection ===

    loss_dict = {
        'sse': SSELoss,
        'softmaxce': SoftmaxCELoss,
        'eucEMD': EucEMDLoss,
    }
    
    loss_params = flatten_namespace_to_dict(params.loss)
    loss_keys = params.training.loss
    lambdas = params.training.lambdas

    criteria = []
    for i, lk in enumerate(loss_keys):
        if lk not in loss_dict:
            raise ValueError(f"Unsupported loss function '{lk}'. Choose from: {list(loss_dict.keys())}.")
        # Dynamically instantiate the loss function with parameters.
        loss_cls = loss_dict[lk]
        criterion = loss_cls(_lambda=lambdas[i], **loss_params)
        criteria.append(criterion)

    # === Multinoise Loss Initialisation === 
    if params.training.mnl:
        multi_loss = MultiNoiseLoss(n_losses=len(criteria), device=device).to(device)
    else:
        multi_loss = None
    # === Optimizer Selection ===
    optim_dict = {
        'adam': optim.Adam,
        'sgd': partial(optim.SGD, momentum=params.optimizer.sgd.momentum)
    }
    optim_name = params.training.optimizer
    optimizer_cls = optim_dict.get(optim_name)
    if optimizer_cls is None:
        raise ValueError(f"Unsupported optimizer '{optim_name}'. Choose from: {list(optim_dict.keys())}.")
    # Set learning rate.
    learning_rate = params.optimizer.lr
    if params.training.mnl:
        optimizer = optimizer_cls([
                                    {'params': model.parameters()},
                                    {'params': multi_loss.noise_params}], lr=learning_rate)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=learning_rate)
    
    return model, criteria, optimizer, multi_loss, torch.inf
