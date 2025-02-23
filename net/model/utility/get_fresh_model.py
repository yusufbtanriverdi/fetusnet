import torch.optim as optim
import torch

# Default imports
from net.model.backbone import ResUnet3D  
from net.loss import KLDivergenceLoss, CrossEntropyLoss, MSELoss, EMDLoss


def get_fresh_model(params):
    """Initialize a fresh model, loss function, and optimizer based on params."""
    
    # Load model with default fallbacks
    model = ResUnet3D(
        in_channels=1, 
        out_channels=getattr(params, "num_lmks", 10),  
        num_features=getattr(params, "num_fts", 64)
    )
    model.to(getattr(params, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(model)

    # Loss function mapping
    loss_dict = {
        "MSE": MSELoss,
        "KLD": KLDivergenceLoss,
        "DCE": CrossEntropyLoss,
        "EMD": EMDLoss
    }
    
    # Set loss function with fallback
    loss_name = getattr(params, "loss", "MSE")  # Default to "MSE"
    criterion = loss_dict.get(loss_name, MSELoss)()

    # Optimizer mapping
    optim_dict = {
        "Adam": optim.Adam,
        "SGD": optim.SGD
    }

    # Get optimizer with default fallback
    optim_name = getattr(params, "optim", "Adam")  # Default to "Adam"
    optimizer_cls = optim_dict.get(optim_name)  

    if optimizer_cls is None:
        raise ValueError(f"Unsupported optimizer '{optim_name}'. Choose from {list(optim_dict.keys())}.")
    
    optimizer = optimizer_cls(model.parameters(), lr=getattr(params, "lr", 1e-3))

    return model, criterion, optimizer, torch.inf
