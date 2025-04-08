import torch.optim as optim
import torch
from functools import partial

# Default imports
from net.model.backbone import ResUNet3D  
from net.loss.voxel.mean_squared_error import MSELoss as voxelMSE
from net.loss.voxel.kullback_leibler_div import KLDLoss as voxelKLD
from net.loss.voxel.distance_matrix_multiplication import DMLoss as voxelDM
from net.loss.voxel.emd import EMDLoss as voxelEMD
from net.loss.joint.joint_histogram import JointHistogramLoss as joint
from net.loss.histogram.total_variation_distance import TVDLoss as histMSE


def get_fresh_model(params):
    """Initialize a fresh model, loss function, and optimizer based on params."""
    
    # Load model with default fallbacks
    model = ResUNet3D.ResUNet3D(
        input_channels=1, 
        output_channels=len(params.lmks),  
        base_features=getattr(params, "num_fts", 64)
    )
    model.to(getattr(params, "device", "cuda" if torch.cuda.is_available() else "cpu"))

    loss_dict = {
        'joi': partial(joint, reduction=getattr(params, "reduction", 'mean'), bins=params.n_bins, sigma=params.sigma),  # :(
        'tvd': partial(histMSE, reduction=getattr(params, "reduction", 'mean'), bins=params.n_bins), # ?   :()
        'kld': partial(voxelKLD, reduction=getattr(params, "reduction", 'mean')),  # + 
        'dml': partial(voxelDM, reduction=getattr(params, "reduction", 'mean')), 
        'mse': partial(voxelMSE, reduction=getattr(params, "reduction", 'mean'), bins=params.n_bins),   # + 
        'emd': partial(voxelEMD, reduction=getattr(params, "reduction", 'mean')),  # + 
        }
   
    # Set loss function with fallback
    loss_name = getattr(params, 'loss', 'mse')  # Default to mse
    if loss_name in loss_dict:
        criterion = loss_dict[loss_name]()  # Call the function to create an instance
    else:
        raise ValueError(f"Unsupported loss function '{loss_name}'. Choose from {list(loss_dict.keys())}.")

    # Optimizer mapping
    optim_dict = {
        'adam': optim.Adam,
        'sgd': partial(optim.SGD, momentum=0.3) 
    }
    # Get optimizer with default fallback
    optim_name = getattr(params, 'optimizer', 'adam')  # Default to adam

    optimizer_cls = optim_dict.get(optim_name)  

    if optimizer_cls is None:
        raise ValueError(f"Unsupported optimizer '{optim_name}'. Choose from {list(optim_dict.keys())}.")
    
    optimizer = optimizer_cls(model.parameters(), lr=getattr(params, 'learning_rate', 1e-3))
    
    print(model, criterion, optimizer)

    return model, criterion, optimizer, torch.inf