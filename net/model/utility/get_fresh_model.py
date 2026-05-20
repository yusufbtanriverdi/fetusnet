import torch.optim as optim
import torch
from functools import partial

from net.model.backbone import ResUNet3D
from net.loss.losses import *

def get_fresh_model(params):
    """
    Initialize a fresh model, loss function, and optimizer based on the provided parameters.

    This function sets up:
      - A 3D CNN model for medical image processing or landmark detection tasks.
      - A loss function chosen from several custom or standard options.
      - An optimizer configured with the desired learning rate and settings.

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
    model_key = getattr(params, 'architecture', 'resunet3d')  # Default to 'mse' if not specified.
    
    if model_key == 'resunet3d':
        model = ResUNet3D.ResUNet3D(
            input_channels=1,  # Assuming single-channel input (e.g., grayscale or single-modality volumes).
            output_channels=len(params.lmks),  # Output channels match the number of landmarks.
            base_features=getattr(params, "num_fts", 32),  # Base feature count (default: 32).
            coord_reg=getattr(params, "coord_reg", False)  # Whether to use coordinate regression.
        )
    else: 
        raise ValueError(f"Unsupported model architecture '{model_key}'. Choose from: ['resunet3d', 'resnet34'].")
    
    # Move model to the specified device (default to CUDA if available).
    device = getattr(params, "device", "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # === Loss Function Selection ===
    reduction = getattr(params, "reduction", 'mean')
    loss_dict = {
        'sse': SSELoss,
        'softmaxce': SoftmaxCELoss,
        'eucEMD': EucEMDLoss,
    }
    loss_keys = getattr(params, 'loss', ['softmaxce'])  # Default to 'mse' if not specified.
    lambdas = getattr(params, 'lambdas', [1.0]) 
    criteria = []
    for i, lk in enumerate(loss_keys):
        if lk not in loss_dict:
            raise ValueError(f"Unsupported loss function '{lk}'. Choose from: {list(loss_dict.keys())}.")

        # Dynamically instantiate the loss function with parameters.
        loss_cls = loss_dict[lk]
        loss_params = getattr(params, 'loss_params', {}) # Additional parameters for the loss function.
        criterion = loss_cls(reduction=reduction, _lambda=lambdas[i], **loss_params)
        criteria.append(criterion)

    # === Multinoise Loss Initialisation === 
    if params.mnl:
        multi_loss = MultiNoiseLoss(n_losses=len(criteria)).to(device)
    else:
        multi_loss = None
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
    # Set learning rate.
    learning_rate = getattr(params, 'learning_rate', 1e-3)
    if params.mnl:
        optimizer = optimizer_cls([
                                    {'params': model.parameters()},
                                    {'params': multi_loss.noise_params}], lr=learning_rate)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=learning_rate)
    
    return model, criteria, optimizer, multi_loss, torch.inf
