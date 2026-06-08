import datetime
import os
import wandb
import json

def create_experiment_id(params, create_directory=True):
    """
    Creates a unique experiment directory under runs/YYYY-MM-DD/, named as prefix + counter suffix.
    
    Args:
        params: Object with attribute `prefix` (experiment base name).
        create_directory (bool): If True, creates the directory.
    
    Returns:
        tuple:
            - experiment_directory (str or None): Full path like 'runs/YYYY-MM-DD/prefix_counter'
            - experiment_id (str): Unique experiment name like 'prefix' or 'prefix_1', 'prefix_2', etc.
    """

    current_time = datetime.datetime.now()
    date_str = current_time.strftime("%Y-%m-%d")
    base_dir = os.path.join("runs", date_str)
    os.makedirs(base_dir, exist_ok=True)

    base_experiment_name = params.prefix
    experiment_name = base_experiment_name
    experiment_dir = os.path.join(base_dir, experiment_name)
    counter = 1

    while os.path.exists(experiment_dir):
        experiment_name = f"{base_experiment_name}_{counter}"
        experiment_dir = os.path.join(base_dir, experiment_name)
        counter += 1

    if create_directory:
        os.makedirs(experiment_dir, exist_ok=False)

    return experiment_dir, experiment_name

def save_experiment_parameters(experiment_directory, experiment_id, params, date):
    """
    Saves experiment parameters and metadata as a JSON file inside the experiment directory.

    Args:
        experiment_directory (str): Path to the experiment directory where the log will be saved.
        experiment_id (str): Unique experiment identifier.
        params (object or dict): Parameters to log. Can be a Namespace, object with __dict__, or dict.
        date (str): Date string of the experiment (e.g., '2025-07-03').
    """

    # Ensure the experiment directory exists
    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory, exist_ok=True)

    log_file = os.path.join(experiment_directory, "parameters.json")

    # Convert params to dictionary safely
    if hasattr(params, '__dict__'):
        params_dict = vars(params)
    elif isinstance(params, dict):
        params_dict = params
    else:
        # Try to convert any other object with attributes
        params_dict = {k: getattr(params, k) for k in dir(params) if not k.startswith('_') and not callable(getattr(params, k))}

    # Add experiment ID and date into the log data
    log_data = {
        "Experiment_ID": experiment_id,
        "Date": date,
        "Parameters": params_dict
    }

    # Write to JSON file with indentation for readability
    with open(log_file, "w") as log:
        json.dump(log_data, log, indent=4)

def log_epoch_to_wandb(train_loss, val_loss, ep_scores, params, global_wandb_steps):
    """
    Logs training and validation losses along with landmark-specific metrics to Weights & Biases (wandb).

    Args:
        train_loss (float): Training loss for the current epoch.
        val_loss (float): Validation loss for the current epoch.
        ep_scores (dict): Dictionary containing evaluation metrics (e.g., dmean per landmark).
        params: Configuration object containing:
            - use_wandb (bool): Flag to enable wandb logging.
            - lmks (list): List of landmark identifiers.
        global_wandb_steps (dict): Dictionary tracking global step counters (e.g., {'epoch': int}).
    """
    if not getattr(params, 'use_wandb', False):
        return  # Do nothing if wandb is disabled

    # Log generic losses
    wandb.log({'epoc/val_loss': val_loss, 'epoc/epoch': global_wandb_steps['epoch']})
    wandb.log({'epoc/train_loss': train_loss, 'epoc/epoch': global_wandb_steps['epoch']})
    wandb.log({'epoc/dmean': ep_scores['dmean'].mean(), 'epoc/epoch': global_wandb_steps['epoch']})

    # Log per-landmark evaluation metrics
    # for lmk in params.lmks:
    #     metric_name = f'dmean_{lmk}'
    #     if metric_name in ep_scores:
    #         wandb.log({f'epoc/{metric_name}': ep_scores[metric_name].mean(),
    #                    'epoc/epoch': global_wandb_steps['epoch']})

    # Increment epoch counter
    global_wandb_steps['epoch'] += 1
    return global_wandb_steps

def update_dataframe(dataframe, params):
    # Construct full file paths by joining base paths with relative paths
    paths = dataframe['mscan'].apply(lambda x: os.path.join(params.sys + params.root, x))

    # Create a boolean mask for existing files
    mask = paths.apply(os.path.exists)

    # Filter DataFrame by mask and reset index
    dataframe = dataframe[mask].reset_index(drop=True)
    
    # Clear nonfrontal
    dataframe = dataframe[~dataframe['nonfrontal_after_rot']].reset_index(drop=True)
    # Clear set = -1 
    dataframe = dataframe[dataframe['set'] != -1].reset_index(drop=True)
    
    return dataframe

def test_loaders(dataloader):
    from tqdm import tqdm
    """
    Test function to validate the DataLoader functionality.

    This function iterates through the provided DataLoader, printing out the shapes
    of the input volumes and target heatmaps for each batch. It serves as a basic
    sanity check to ensure that the DataLoader is correctly loading and batching data.

    Args:
        dataloader (torch.utils.data.DataLoader): The DataLoader instance to be tested.

    Returns:
        None
    """
    for _ in tqdm(dataloader):
        pass
    return
