import os
import pandas as pd
import torchio as tio
import warnings
import wandb

# Import project modules
from net.config.wandb import initialize_wandb
from net.dataset import generate_targets
from net.dataset.utility.loaders import get_train_val_dl, get_test_dl
from net.config.create_experiment_id import create_experiment_id
from net.dataset.statistics import create_info_frames, split_patient_fold
from net.dataset import rotate
from net.parameters.parameters import parameters_parsing
from net.train import train_one_ep
from net.infer import infer_one_ep
from net.model.utility.checkpoints import load_checkpoint, save_checkpoint
from net.model.utility.get_fresh_model import get_fresh_model


def train_and_validate_one_fold(lmk, fold, params, transformations, experiment_directory, global_wandb_steps):
    print(f"\n--- Training Fold {fold + 1}/{params.n_split} ---")

    if params.use_wandb:
        initialize_wandb(params, fold=fold)
        
    # Initialize model, loss, optimizer
    model, criterion, optimizer, best_criteria = get_fresh_model(params)
    if params.resume:
        experiment_directory = params.exp_dir
        model_dir = f'{experiment_directory}/last_fold{fold}.pt'
        model, optimizer, epoch, best_criteria = load_checkpoint(model, optimizer, model_dir)
    else:
        epoch = 0
        best_criteria = float('inf')
    print(params)
    # Load training and validation data
    train_dl, val_dl = get_train_val_dl(lmk, fold, params, transformations=transformations)

    # Epoch training loop
    for epoch in range(epoch, params.epochs):
        print(f"\n[Epoch {epoch + 1}/{params.epochs}]")

        # Train for one epoch
        train_loss, global_wandb_steps = train_one_ep(
            model, train_dl, criterion, optimizer, params.device, 
            wandb_steps=global_wandb_steps,
            use_wandb=params.use_wandb
        )

        # Validate for one epoch
        val_loss, ep_scores, global_wandb_steps = infer_one_ep(
            model, val_dl, criterion, params.device, 
            wandb_steps=global_wandb_steps,
            use_wandb=params.use_wandb
        )

        if params.use_wandb:
            # Log metrics to WandB
            wandb.log({'epoc/val_loss': val_loss, 'epoc/epoch': global_wandb_steps['epoch']})
            wandb.log({'epoc/train_loss': train_loss, 'epoc/epoch': global_wandb_steps['epoch']})

        # Save best model
        if val_loss < best_criteria:
            best_criteria = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, 
                            os.path.join(experiment_directory, f'best_fold{fold}.pt'))

        # Save last model
        save_checkpoint(model, optimizer, epoch, val_loss, 
                        os.path.join(experiment_directory, f'last_fold{fold}.pt'))

        print("Last Ep d-mean Score: ", ep_scores['dmean'].item())

    # Initialize model, loss, optimizer
    model, criterion, optimizer, best_criteria = get_fresh_model(params)
    # Load the best model checkpoint
    model_dir = f'{experiment_directory}/best_fold{fold}.pt'
    model, optimizer, epoch, best_val_loss = load_checkpoint(model, optimizer, model_dir)
    print(f"Best validation loss: {best_val_loss}") 

    # Final evaluation on validation set
    val_loss, ep_scores, global_wandb_steps = infer_one_ep(
        model, val_dl, criterion, params.device, 
        wandb_steps=global_wandb_steps, eval=True, save_dir=experiment_directory
    )

    if params.use_wandb:
        # Finish WandB logging
        wandb.finish()

    return global_wandb_steps, best_val_loss

# Suppress UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize parameters
params = parameters_parsing()

# Create experiment directory and log
experiment_directory, experiment_id = create_experiment_id(params)

# Initialize global tracking dictionary for WandB
global_wandb_steps = {'train_loss': 0, 'val_loss': 0, 'epoch': 0}

# Create local experiment log
log_file = os.path.join(experiment_directory, "experiment_log.txt")
with open(log_file, "w") as log:
    log.write(f"Experiment ID: {experiment_id}\n")
    log.write(f"Parameters: {vars(params)}\n")

# Set system paths based on OS
if params.os in ['linux', 'l']:
    params.sys = "/media/yusuf/HDD 4TB/"
else:
    params.sys = "D:/"

print(f"System path set to: {params.sys}")
print('Device: ', params.device)

# Handle specific modes for data preparation
if params.mode in ['script_prepare', 'script_split']:
    print(f" WELCOME TO {params.mode} MODE. \nCreating info frames...")
    sinfo_df = create_info_frames.main(params)

if params.mode == 'script_split':
    print(f" WELCOME TO {params.mode} MODE. \nSplitting patients into train-test sets...")
    sinfo_df = split_patient_fold.main(sinfo_df, params)

# Load or create sinfo dataframe
sinfo_path = params.sys + params.root + 'sinfo.csv'
print(f"Loading sinfo from {sinfo_path}...")
if os.path.exists(sinfo_path):
    sinfo_df = pd.read_csv(sinfo_path)
else:
    raise FileNotFoundError("sinfo.csv not found.")

# Filter and update sinfo dataframe
sinfo_df = sinfo_df[sinfo_df['landmark_antonia_found']].reset_index(drop=True)
print(len(sinfo_df))
sinfo_df = create_info_frames.update(sinfo_df, params)
print(len(sinfo_df))

# Rotate/alignment step (commented out by default)
# if params.mode == 'script_rotate':
#     rotate.main(sinfo_df, params)

# Generate target datasets
if params.mode == 'script_generate_targets':
    print(params.sys)
    generate_targets.main(sinfo_df, experiment_directory, params)

# Define transformations for 3D images
transforms = [tio.RescaleIntensity((0, 1))] if params.rescale else []
transformations = tio.Compose(transforms)

# Training phase
if params.mode in ['train', 'train_test']:
    # Ensure supported training mode
    if params.training_mode != 'one-by-one':
        raise ValueError(f"Unsupported execution mode: {params.training_mode}. "
                         "Multiple landmarks cause high computation costs.")

    # Loop through landmarks
    for lmk in params.lmks:
        # Training across folds
        if not params.iter_folds:
            iter_folds = range(params.n_split)
        else:
            iter_folds = params.iter_folds

        for fold in iter_folds:
            # Initialize global tracking dictionary for WandB
            global_wandb_steps = {'train_loss': 0, 'val_loss': 0, 'epoch': 0}
            global_wandb_steps, best_val_loss = train_and_validate_one_fold(lmk, fold, params, transformations, experiment_directory, global_wandb_steps)

    with open(log_file, "w") as log:
        log.write(f"Experiment ID: {experiment_id}\n")
        log.write(f"Parameters: {vars(params)}\n")
        log.write(f"Training completed for all folds.\n")
        log.write(f"Best validation loss: {best_val_loss}\n")   


# Testing or evaluation phase
if params.mode in ['test', 'eval']:
    # Ensure supported execution mode
    if params.training_mode != 'one-by-one':
        raise ValueError(f"Unsupported execution mode: {params.execution}. "
                         "Multiple landmarks cause high computation costs.")

    # Loop through landmarks
    for lmk in params.lmks:
        # Initialize model, loss, optimizer
        model, criterion, optimizer, best_criteria = get_fresh_model(params)
        # Load the best model checkpoint
        model_dir = f'runs/{params.model_dir}/best_fold{params.test_fold}.pt'
        model, optimizer, epoch, best_val_loss = load_checkpoint(model, optimizer, model_dir)

        # Load test data
        test_dl = get_test_dl(params, lmk, transformations=transformations)

        # Final evaluation on test set
        infer_one_ep(
            model, test_dl, criterion, params.device, 
            wandb_steps=global_wandb_steps, eval=True, save_dir=experiment_directory, use_wandb=params.use_wandb
        )


# Experiments to be done_
# python fetusnet.py train --ep 100 --wandbpro fetusnetv1 --run_name softmax_crosse_adam --optim adam --num_fts 32 --lr 0.0001 --loss sce --use_wandb --iter_folds 0 1 2 3
