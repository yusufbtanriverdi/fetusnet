import torch.optim as optim
import torch
from functools import partial

# Importing necessary modules for the model, loss functions, and optimizer
from net.model.backbone import ResUNet3D  
from net.loss.voxel.v1.mean_squared_error import MeanSquaredErrorLoss
from net.loss.voxel.v1.kullback_leibler_div import KullbackLeiblerDivLoss
from net.loss.voxel.kullback_leibler_div import KullbackLeiblerDivLossV2
from net.loss.voxel.softmax_cross_entropy import SoftmaxCrossEntropyLoss
from net.loss.voxel.emd_regularization import DistanceMatrixLoss
from net.loss.utils import CombinedLoss

def get_fresh_model(params):
    """
    Initialize a fresh model, loss function, and optimizer based on the provided parameters.

    Args:
        params: An object containing configuration parameters such as:
            - lmks: List of landmarks (used to determine output channels).
            - num_fts: Number of base features for the model (default: 64).
            - device: Device to use ('cuda' or 'cpu').
            - reduction: Reduction method for loss functions (default: 'mean').
            - loss: Name of the loss function to use (default: 'mse').
            - optimizer: Name of the optimizer to use (default: 'adam').
            - learning_rate: Learning rate for the optimizer (default: 1e-3).

    Returns:
        model: The initialized model.
        criterion: The selected loss function.
        optimizer: The selected optimizer.
        torch.inf: A placeholder value (can be replaced with a more meaningful return if needed).
    """
    
    # Initialize the model with default or user-specified parameters
    model = ResUNet3D.ResUNet3D(
        input_channels=1,  # Assuming single-channel input
        output_channels=len(params.lmks),  # Number of output channels based on landmarks
        base_features=getattr(params, "num_fts", 64)  # Default base features: 64
    )
    # Move the model to the specified device (default: 'cuda' if available, else 'cpu')
    model.to(getattr(params, "device", "cuda" if torch.cuda.is_available() else "cpu"))

    loss_dict = {
        # Version 1 loss functions
        'kld': partial(KullbackLeiblerDivLoss, reduction=getattr(params, "reduction", 'mean')),  
        'kldv2': partial(KullbackLeiblerDivLossV2, reduction=getattr(params, "reduction", 'mean')),
        'mse': partial(MeanSquaredErrorLoss, reduction=getattr(params, "reduction", 'mean')),   
        # Version 2 loss functions
        'sce': partial(SoftmaxCrossEntropyLoss, reduction=getattr(params, "reduction", 'mean')), 
        'dis': partial(DistanceMatrixLoss, reduction=getattr(params, "reduction", 'mean')), 
    }
   
    # Select the loss function based on user input or default to 'mse'
    loss_name = getattr(params, 'loss', ['mse'])  # Default: 'mse'

    # If params.loss is a list, handle multiple losses
    if isinstance(loss_name, list):
        if len(loss_name) == 0:
            raise ValueError("The loss list is empty. Please provide at least one loss function.")
        elif len(loss_name) == 1:
                # Handle single loss case
            if loss_name[0] in loss_dict:
                criterion = loss_dict[loss_name[0]]()  # Instantiate the loss function
            else:
                raise ValueError(f"Unsupported loss function '{loss_name[0]}'. Choose from {list(loss_dict.keys())}.")
        else:
            try:
                # If multiple losses are provided, combine them into a single loss function
                criterion = CombinedLoss(
                    [loss_dict[loss]() for loss in loss_name], 
                    weights=[1.0 for _ in loss_name],  # Equal weights for each loss 
                    reduction=getattr(params, "reduction", 'mean')
                )
                print(criterion)
            except KeyError as e:
                raise ValueError(f"Unsupported loss function '{e.args[0]}'. Choose from {list(loss_dict.keys())}.")

    print(f"Using loss function: {loss_name}")
    if 'dis' in [loss_name]:
       params.dist_matrix = True  # Ensure distance matrix is computed for EMD loss
    else:
        # If not using EMD loss, ensure the model does not output a distance matrix
        params.dist_matrix = False  # Ensure distance matrix is not computed
    # Define a dictionary of available optimizers
    optim_dict = {
        'adam': optim.Adam,  # Adam optimizer
        'sgd': partial(optim.SGD, momentum=0.9)  # SGD optimizer with momentum
    }
    
    # Select the optimizer based on user input or default to 'adam'
    optim_name = getattr(params, 'optimizer', 'adam')  # Default: 'adam'
    optimizer_cls = optim_dict.get(optim_name)  # Get the optimizer class

    if optimizer_cls is None:
        raise ValueError(f"Unsupported optimizer '{optim_name}'. Choose from {list(optim_dict.keys())}.")
    
    # Instantiate the optimizer with model parameters and learning rate
    optimizer = optimizer_cls(model.parameters(), lr=getattr(params, 'learning_rate', 1e-3))
    
    # Print the model, loss function, and optimizer for debugging purposes
    print(model, criterion, optimizer)

    # Return the initialized components
    return model, criterion, optimizer, torch.inf