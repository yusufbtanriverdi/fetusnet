import os
import pandas as pd
import torchio as tio
import warnings
import wandb
import json
import torch

from net.parameters.parameters import parameters_parsing
from net.config.create_experiment_id import create_experiment_id
from net.dataset.statistics.split import perform_split
from net.dataset.statistics.prepare import perform_prepare
from net.dataset.rotate import perform_rotate
from net.dataset.generate_targets import perform_generate
from logger_setup import setup_logger
from net.dataset.utility.loaders import get_train_val_dl, get_test_dl
from net.model.utility.get_fresh_model import get_fresh_model
from net.phases.train import train_one_ep
from net.phases.infer import infer_one_ep
from net.model.utility.checkpoints import load_checkpoint, save_checkpoint
from net.config.wandb import initialize_wandb

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
    wandb.log({'epoc/dmean': ep_scores['dmean'].item(), 'epoc/epoch': global_wandb_steps['epoch']})

    # Log per-landmark evaluation metrics
    # for lmk in params.lmks:
    #     metric_name = f'dmean_{lmk}'
    #     if metric_name in ep_scores:
    #         wandb.log({f'epoc/{metric_name}': ep_scores[metric_name].mean(),
    #                    'epoc/epoch': global_wandb_steps['epoch']})

    # Increment epoch counter
    global_wandb_steps['epoch'] += 1
    return global_wandb_steps

def train_and_validate(fold_dataframe, fold_experiment_dir, params, transformations, global_wandb_steps):
    train_dl, val_dl = get_train_val_dl(fold_dataframe, params, transformations=transformations)

    # Initialize model, loss, optimizer
    model, criterion, optimizer, best_criteria = get_fresh_model(params)
    logger.info(optimizer)
    logger.info(criterion)

    train_losses = []
    val_losses = []
    # Epoch training loop
    for epoch in range(0, params.epochs):
        logger.info(f" \n[Epoch {epoch + 1}/{params.epochs}] \n ")

        # Train for one epoch
        train_loss, global_wandb_steps = train_one_ep(
            model, 
            train_dl, 
            criterion, 
            optimizer, 
            params.device, 
            global_wandb_steps,
            params.use_wandb,
            progress_bar=params.progress_bar
        )

        val_loss, global_wandb_steps, ep_scores = infer_one_ep(
            model, 
            val_dl, 
            criterion, 
            params.device, 
            global_wandb_steps, 
            params.use_wandb,
            params.detector, 
            params.progress_bar,     
            params.lmks, 
            experiment_dir,
            eval=False, 
        )
        
        # Save best model
        if val_loss < best_criteria:
            best_criteria = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, 
                            os.path.join(fold_experiment_dir, f'best.pt'))

        # Save last model
        save_checkpoint(model, optimizer, epoch, val_loss, 
                        os.path.join(fold_experiment_dir, f'last.pt'))
        logger.info(f"Train loss: {train_loss}")
        logger.info(f"Validation loss: {val_loss}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        global_wandb_steps = log_epoch_to_wandb(train_loss, val_loss, ep_scores, params, global_wandb_steps)

    return train_losses, val_losses

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
    for batch in tqdm(dataloader):
        pass
    return

print("Torch cuda is available? ", torch.cuda.is_available())
# Suppress UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
# Initialize parameters
params = parameters_parsing()
experiment_dir, experiment_name = create_experiment_id(params, create_directory=True)
# Convert to dictionary (if Namespace or similar)
params_dict = vars(params)  # or: params.__dict__ if vars() doesn't work

# Write to JSON
with open(os.path.join(experiment_dir, 'params.json'), 'w') as f:
    json.dump(params_dict, f, indent=6)
    
global_wandb_steps = {'train_loss': 0, 'val_loss': 0, 'epoch': 0}
# Initialize logger
logger = setup_logger(experiment_dir)
logger.info(f"Experiment started and parameters are saved. This is the experiment dir: {experiment_dir} ")
logger.info(f"System path set to: {params.sys}")
logger.info(f"Device set to: {params.device}")
logger.info("Searching for master dataframe in system+dataset_root... \n If you did not prepare such dataframe or you think it misses some samples, please re-run this script in prepare mode and specify datasets")

# Create info frames
if params.mode == 'prepare':
    master_dataframe_path = perform_prepare(params)

# Load or create sinfo dataframe
master_dataframe_path = os.path.join(params.sys, params.root, params.master_df + '.csv')
if os.path.exists(master_dataframe_path):
    master_dataframe = pd.read_csv(master_dataframe_path)
else:
    logger.error(f"Master dataframe not found @ {master_dataframe_path}.")
    raise FileNotFoundError("Master dataframe not found.")

# Rotate/alignment step
if params.mode == 'rotate':
    perform_rotate(master_dataframe, params)

# Split into subsets
if params.split != 'crossfold':
    split_dataframe_path = os.path.join(params.sys, params.root, params.master_df + params.split + '.csv')
else: # Takes the first number as fold in iter folds as default ! BE CAREFUL WITH TEST.
    logger.info(f"Crossfold detected. You selected folds: {params.iter_folds}")
    split_dataframe_path = os.path.join(params.sys, params.root, params.master_df + f'_fold{params.iter_folds[0]}.csv')

if params.mode == 'presplit':
    split_dataframe_path = perform_split(master_dataframe, params)

logger.info(f"Master dataframe path: {master_dataframe_path}")
logger.info(f"Split dataframe path: {split_dataframe_path}")    
logger.info(f"Total number of input in master dataframe: {len(master_dataframe)}")
logger.info(f"Moving into splitting... You selected {params.split}")
logger.info("I am checking if there is already a split dataframe for this splitter...")
if os.path.exists(split_dataframe_path):
    splitted_dataframe = pd.read_csv(split_dataframe_path)
else:
    logger.warning("Split dataframe not found. Please re-run this script in presplit mode to store splitting information. Now, I will split for you. \n")
    split_dataframe_path = perform_split(master_dataframe, params)
    splitted_dataframe = pd.read_csv(split_dataframe_path)


logger.info(f"Training - validation - test subset sizes: {len(splitted_dataframe[splitted_dataframe['set'] == 0]), len(splitted_dataframe[splitted_dataframe['set'] == 1]), len(splitted_dataframe[splitted_dataframe['set'] == 2])}")

# Define transformations for 3D images
transforms = [
                tio.RescaleIntensity((0, 1)), 
                # tio.Flip(axes=('L',)),
              ] if params.rescale else []
transformations = tio.Compose(transforms)
logger.info("Transformations are created.")


if params.mode == 'pipe':
    print(splitted_dataframe[:10])
    train_dl, test_dl = get_train_val_dl(splitted_dataframe, params, transformations=transformations)
    test_loaders(train_dl)
    test_loaders(test_dl)

logger.info(f"I am cleaning the dataframe from non-existing files and nonfrontal cases... {len(splitted_dataframe)}")
splitted_dataframe = update_dataframe(splitted_dataframe, params)
logger.info(f"After cleaning, total number of input in master dataframe: {len(splitted_dataframe)}")
logger.info(f"!__[U]__! Training - validation - test subset sizes: {len(splitted_dataframe[splitted_dataframe['set'] == 0]), len(splitted_dataframe[splitted_dataframe['set'] == 1]), len(splitted_dataframe[splitted_dataframe['set'] == 2])}")

if params.mode == 'generate':
    perform_generate(splitted_dataframe, experiment_dir, params)

if params.mode == 'train':
    logger.info("I am starting to train")
    # This comes with validation.
    if 'crossfold' in params.split:
        for fold in params.iter_folds:

            split_dataframe_path = os.path.join(params.sys, params.root, params.master_df + f'_fold{fold}.csv')
            splitted_dataframe = pd.read_csv(split_dataframe_path) 
            logger.info(f"I am cleaning the dataframe from non-existing files and nonfrontal cases... {len(splitted_dataframe)}")
            splitted_dataframe = update_dataframe(splitted_dataframe, params)
            logger.info(f"After cleaning, total number of input in master dataframe: {len(splitted_dataframe)}")
            logger.info(f"!__[U]__! Training - validation - test subset sizes: {len(splitted_dataframe[splitted_dataframe['set'] == 0]), len(splitted_dataframe[splitted_dataframe['set'] == 1]), len(splitted_dataframe[splitted_dataframe['set'] == 2])}")

            if params.use_wandb:
                initialize_wandb(params, experiment_name+'_fold' + str(fold))

            fold_experiment_dir = os.path.join(experiment_dir, f'fold_{fold}')
            os.makedirs(fold_experiment_dir, exist_ok=True)
            logger.info(f"\n--- Training Fold {fold}/{params.n_split-1} --- \n")
            # Initialize model, loss, optimizer
            model, criterion, optimizer, best_criteria = get_fresh_model(params)
            # If given, load pretrained weights.
            # TODO
            train_losses, val_losses = train_and_validate(splitted_dataframe, fold_experiment_dir, params, transformations, global_wandb_steps)
            # Combine losses into a DataFrame
            loss_df = pd.DataFrame({
                'epoch': list(range(len(train_losses))),
                'train_loss': train_losses,
                'val_loss': val_losses
            })
            # Save to CSV or any preferred format
            loss_df.to_csv(os.path.join(fold_experiment_dir, f"fold_{fold}_losses.csv"), index=False)

            model_dir = f'{fold_experiment_dir}/best.pt'
            model, optimizer, epoch, best_val_loss = load_checkpoint(model, optimizer, model_dir)
            test_dl = get_test_dl(splitted_dataframe, params, transformations=transformations)
            if len(test_dl) == 0: _, test_dl = get_train_val_dl(splitted_dataframe, params, transformations=transformations)
            test_loss, global_wandb_steps, test_scores = infer_one_ep(
                model, 
                test_dl, 
                criterion, 
                params.device, 
                global_wandb_steps, 
                params.use_wandb, 
                params.detector, 
                params.progress_bar,     
                params.lmks, 
                experiment_dir,
                eval=True,
                radius_eval=params.radius_eval,
                radius_num=params.radius_num,
                save_targets=False, 
                save_outputs=False 
                )
            
            for lmk in params.lmks:
                mean = test_scores[f"dmean_{lmk}"].mean()
                std = test_scores[f"dmean_{lmk}"].std()
                logger.info(f"Test d-mean Score for {lmk} @ fold {fold}: {mean} +/- {std}")
            
            if params.use_wandb: wandb.finish()
            global_wandb_steps = {'train_loss': 0, 'val_loss': 0, 'epoch': 0}
            
    else:
        if params.use_wandb: initialize_wandb(params, experiment_name+'_'+params.split)
        logger.info(f"\n--- Training {params.split} --- \n")
        # Initialize model, loss, optimizer
        model, criterion, optimizer, best_criteria = get_fresh_model(params)
        train_losses, val_losses = train_and_validate(splitted_dataframe, experiment_dir, params, transformations, global_wandb_steps)
        # Combine losses into a DataFrame
        loss_df = pd.DataFrame({
            'epoch': list(range(len(train_losses))),
            'train_loss': train_losses,
            'val_loss': val_losses
        })
        # Save to CSV or any preferred format
        loss_df.to_csv(os.path.join(experiment_dir, f"losses.csv"), index=False)

        model_dir = f'{experiment_dir}/best.pt'
        model, optimizer, epoch, best_val_loss = load_checkpoint(model, optimizer, model_dir)        
        test_dl = get_test_dl(splitted_dataframe, params, transformations=transformations)
        if len(test_dl) == 0: _, test_dl = get_train_val_dl(splitted_dataframe, params, transformations=transformations)
        test_loss, global_wandb_steps, test_scores  = infer_one_ep(
                model, 
                test_dl, 
                criterion, 
                params.device, 
                global_wandb_steps, 
                params.use_wandb, 
                params.detector, 
                params.progress_bar,     
                params.lmks, 
                experiment_dir,
                eval=True,
                radius_eval=params.radius_eval,
                radius_num=params.radius_num,
                save_targets=False, 
                save_outputs=False 
                )
        
        for lmk in params.lmks:
            mean = test_scores[f"dmean_{lmk}"].mean()
            std = test_scores[f"dmean_{lmk}"].std()
            logger.info(f"Test d-mean Score for {lmk}: {mean} +/- {std}")

        if params.use_wandb: wandb.finish()

if params.mode == 'test':
    test_dl = get_test_dl(splitted_dataframe, params, transformations=transformations)
    if len(test_dl) == 0: _, test_dl = get_train_val_dl(splitted_dataframe, params, transformations=transformations)
    # Initialize model, loss, optimizer
    model, criterion, optimizer, best_criteria = get_fresh_model(params)
    model, optimizer, epoch, best_val_loss = load_checkpoint(model, optimizer, params.model_dir)
    test_loss, global_wandb_steps, test_scores = infer_one_ep(
            model, 
            test_dl, 
            criterion, 
            params.device, 
            global_wandb_steps, 
            False, 
            params.detector, 
            params.progress_bar,     
            params.lmks, 
            experiment_dir,
            eval=True, 
            radius_eval=params.radius_eval,
            radius_num=params.radius_num,
            save_targets=False, 
            save_outputs=False
        )

    for lmk in params.lmks:
        mean = test_scores[f"dmean_{lmk}"].mean()
        std = test_scores[f"dmean_{lmk}"].std()
        logger.info(f"Test d-mean Score for {lmk}: {mean} +/- {std}")
