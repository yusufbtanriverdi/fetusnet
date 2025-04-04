import torch.optim as optim
import torch
from functools import partial

# Default imports
from net.model.backbone import ResUNet3D  
from net.loss.voxel.mean_squared_error import MSELoss
from net.loss.voxel.kullback_leibler_div import KLDLoss
from net.loss.joint.total_variation_distance import TVDLoss as TVDJoint
from net.loss.histogram.total_variation_distance import TVDLoss as TVDHist

def get_fresh_model(params):
    """Initialize a fresh model, loss function, and optimizer based on params."""
    
    # Load model with default fallbacks
    model = ResUNet3D.ResUNet3D(
        input_channels=1, 
        output_channels=len(params.lmks),  
        base_features=getattr(params, "num_fts", 64)
    )
    model.to(getattr(params, "device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Set loss function with fallback
    cost_name = getattr(params, 'cost', 'voxel')  # Default to mse

    if cost_name == 'voxel':
        # Loss function mapping with optional parameters
        loss_dict = {
            'mse': partial(MSELoss, reduction=getattr(params, "reduction", 'mean')),  
            'kld': partial(KLDLoss, reduction=getattr(params, "reduction", 'mean')), 
        }

    elif cost_name == 'hist':
            # Loss function mapping with optional parameters
        loss_dict = {
            'mse': partial(TVDHist, reduction=getattr(params, "reduction", 'mean')),  
        }
    
    elif cost_name == 'joint':
             # Loss function mapping with optional parameters
        loss_dict = {
            'mse': partial(TVDJoint, reduction=getattr(params, "reduction", 'mean')),  
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
    optim_name = getattr(params, 'optimizer', 'adam')  # Default to adam
    optimizer_cls = optim_dict.get(optim_name)  

    if optimizer_cls is None:
        raise ValueError(f"Unsupported optimizer '{optim_name}'. Choose from {list(optim_dict.keys())}.")
    
    optimizer = optimizer_cls(model.parameters(), lr=getattr(params, 'learning_rate', 1e-3))
    
    print(model, criterion, optimizer)

    return model, criterion, optimizer, torch.inf