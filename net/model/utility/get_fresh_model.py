import torch.optim as optim
import torch
from functools import partial

# Default imports
from net.model.backbone import ResUNet3D  
from net.loss import KLDivergenceLoss, CrossEntropyLoss, MSELoss, EMDLoss



def get_fresh_model(params):
    """Initialize a fresh model, loss function, and optimizer based on params."""
    
    # Load model with default fallbacks
    model = ResUNet3D.ResUNet3D(
        input_channels=1, 
        output_channels=len(params.lmks),  
        base_features=getattr(params, "num_fts", 64)
    )
    model.to(getattr(params, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(model)

    # Loss function mapping with optional parameters
    loss_dict = {
        'mse': partial(MSELoss.MSELoss, reduction=getattr(params, "reduction", 'mean')),  
        'histmse': partial(MSELoss.HistMSELoss, reduction=getattr(params, "reduction", 'mean'), n_bins=getattr(params, "n_bins", 10)), 
        'jointmse': partial(MSELoss.JointMSELoss, reduction=getattr(params, "reduction", 'mean'), n_bins=getattr(params, "n_bins", 10)), 
        'kld': partial(KLDivergenceLoss.KLDLoss, reduction=getattr(params, "reduction", 'mean')),  
        'histkld': partial(KLDivergenceLoss.HistKLDLoss, reduction=getattr(params, "reduction", 'mean'), n_bins=getattr(params, "n_bins", 10)), 
        'jointkld': partial(KLDivergenceLoss.JointKLDLoss, reduction=getattr(params, "reduction", 'mean'), n_bins=getattr(params, "n_bins", 10)), 

    }
    
    # Set loss function with fallback
    loss_name = getattr(params, 'criterion', 'mse')  # Default to mse
    if loss_name in loss_dict:
        criterion = loss_dict[loss_name]()  # Call the function to create an instance
    else:
        raise ValueError(f"Unsupported loss function '{loss_name}'. Choose from {list(loss_dict.keys())}.")

    # Optimizer mapping
    optim_dict = {
        'adam': optim.Adam,
        'sgd': optim.SGD
    }
    # Get optimizer with default fallback
    optim_name = getattr(params, 'optim', 'adam')  # Default to adam
    optimizer_cls = optim_dict.get(optim_name)  

    if optimizer_cls is None:
        raise ValueError(f"Unsupported optimizer '{optim_name}'. Choose from {list(optim_dict.keys())}.")
    
    optimizer = optimizer_cls(model.parameters(), lr=getattr(params, "lr", 1e-3))

    return model, criterion, optimizer, torch.inf