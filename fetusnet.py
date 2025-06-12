
"""
fetusnet.py
This module serves as the main entry point for the FetusNet project, orchestrating the workflow for training, validating, and testing deep learning models for fetal landmark detection in medical images. It provides functionality for experiment management, data preparation, model training, evaluation, and logging.

Main Functionalities:
---------------------
- Experiment Initialization: Sets up experiment directories, logging, and experiment IDs.
- Data Preparation: Handles creation and updating of patient information frames, dataset splitting, and target generation.
- Model Training & Validation: Trains models for each landmark and fold, supports checkpointing, and logs metrics using Weights & Biases (wandb).
- Model Testing & Evaluation: Loads trained models, evaluates on test data, computes metrics such as Average Expected Local Accuracy (AELA), and saves results.
- Transformations: Applies configurable intensity rescaling and other TorchIO transformations to 3D medical images.
- Configuration: Parses and manages experiment parameters, including device selection, paths, and training modes.

Key Functions:
--------------
- train_and_validate_one_fold: Handles the training and validation loop for a single fold and landmark, including checkpointing and metric logging.
- Main script logic: Manages the workflow based on the selected mode (e.g., data preparation, training, testing).

Usage:
------
Run this script as the main entry point to execute the desired workflow, as specified by the parameters (e.g., mode, etc.).


Note:
-----
- The script assumes the presence of a configuration file and required data files (e.g., sinfo.csv).
- Only 'one-by-one' training mode is supported due to computational constraints.
- Logging and checkpointing are handled per fold and landmark.
"""


import os
import pandas as pd
import torchio as tio
import warnings
import wandb
import torch
import numpy as np

# Import project modules
from net.config.wandb import initialize_wandb
from net.dataset import generate_targets
from net.dataset.utility.loaders import get_train_val_dl, get_test_dl
from net.config.create_experiment_id import create_experiment_id
from net.dataset.statistics import create_info_frames, split_patient_fold
from net.parameters.parameters import parameters_parsing
from net.train import train_one_ep
from net.infer import infer_one_ep
from net.model.utility.checkpoints import load_checkpoint, save_checkpoint
from net.model.utility.get_fresh_model import get_fresh_model
from net.plot.average_expected_local_accuracy import plot_aela_figure

def train_and_validate_one_fold(fold, params, transformations, experiment_directory, global_wandb_steps):

    print(f"\n--- Training Fold {fold + 1}/{params.n_split} ---")

    if params.use_wandb:
        initialize_wandb(params, fold)
        
    # Initialize model, loss, optimizer
    model, criterion, optimizer, best_criteria = get_fresh_model(params)

    if params.resume:
        experiment_directory = params.exp_dir
        model_dir = f'{experiment_directory}/last.pt'
        model, optimizer, in_epoch, best_criteria = load_checkpoint(model, optimizer, model_dir)
    else:
        in_epoch = 0
        best_criteria = float('inf')
    
    print(params)
    # Load training and validation data
    train_dl, val_dl = get_train_val_dl(fold, params, transformations=transformations)

    # Epoch training loop
    for epoch in range(in_epoch, params.epochs):
        print(f"\n[Epoch {epoch + 1}/{params.epochs}]")

        # Train for one epoch
        train_loss, global_wandb_steps = train_one_ep(
            model, train_dl, criterion, optimizer, params.device, 
            wandb_steps=global_wandb_steps,
            use_wandb=params.use_wandb
        )
        # Validate for one epoch
        val_loss, ep_scores, global_wandb_steps = infer_one_ep(
            model, val_dl, criterion, params.device, global_wandb_steps, use_wandb=params.use_wandb, extract_via=params.extract_via, lmks=params.lmks, save_dir=experiment_directory)

        if params.use_wandb:
            # Log metrics to WandB
            wandb.log({'epoc/val_loss': val_loss, 'epoc/epoch': global_wandb_steps['epoch']})
            wandb.log({'epoc/train_loss': train_loss, 'epoc/epoch': global_wandb_steps['epoch']})
            
            for lmk in params.lmks:
                wandb.log({f'epoc/dmean_{lmk}': ep_scores[f'dmean_{lmk}'].mean(), 'epoc/epoch': global_wandb_steps['epoch']})
            global_wandb_steps['epoch'] += 1

        # Save best model
        if val_loss < best_criteria:
            best_criteria = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, 
                            os.path.join(experiment_directory, f'best.pt'))

        # Save last model
        save_checkpoint(model, optimizer, epoch, val_loss, 
                        os.path.join(experiment_directory, f'last.pt'))

    # EVALUATE # 
    # Initialize model, loss, optimizer
    model, criterion, optimizer, best_criteria = get_fresh_model(params)
    # Load the best model checkpoint
    model_dir = f'{experiment_directory}/best.pt'
    model, optimizer, epoch, best_val_loss = load_checkpoint(model, optimizer, model_dir)
    print(f"Best validation loss was {best_val_loss}") 
    # Load test data
    test_dl = get_test_dl(params, fold, transformations=transformations)
    # Final evaluation on test set
    _, ep_scores, global_wandb_steps = infer_one_ep(
        model, test_dl, criterion, params.device, global_wandb_steps, 
        use_wandb=False, 
        extract_via=params.extract_via, 
        eval=True, 
        save_dir=experiment_directory,
        radius_eval=params.radius_eval, 
        radius_num=params.radius_num,  
        lmks=params.lmks,
        )
    
    for lmk in params.lmks:
        print(f"Test d-mean Score for {lmk}: ", ep_scores[f"dmean_{lmk}"].mean(), " +/-", ep_scores[f"dmean_{lmk}"].std())
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
    for k, v in vars(params).items():
        log.write(f"{k}: {v}\n")

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
sinfo_df = create_info_frames.update(sinfo_df, params)

# Rotate/alignment step (commented out by default)
# if params.mode == 'script_rotate':
#     rotate.main(sinfo_df, params)

# Generate target datasets
if params.mode == 'script_generate_targets':
    print(f" WELCOME TO {params.mode} MODE. \n Generating targets...")
    print(params.sys)
    generate_targets.main(sinfo_df, experiment_directory, params)

# Define transformations for 3D images
transforms = [tio.RescaleIntensity((0, 1))] if params.rescale else []
transformations = tio.Compose(transforms)

# Training phase
if params.mode in ['train', 'train_val']:

    # Training across folds
    if not params.iter_folds:
        iter_folds = range(params.n_split)
    else:
        iter_folds = params.iter_folds

    for fold in iter_folds:
            # Create a subdirectory for this fold within the experiment directory
        fold_experiment_dir = os.path.join(experiment_directory, f'fold_{fold}')
        os.makedirs(fold_experiment_dir, exist_ok=True)
        # Initialize global tracking dictionary for WandB
        global_wandb_steps = {'train_loss': 0, 'val_loss': 0, 'epoch': 0}
        global_wandb_steps, best_val_loss = train_and_validate_one_fold(fold, params, transformations, fold_experiment_dir, global_wandb_steps)

        with open(log_file, "a") as log:
            log.write(f"Experiment ID: {experiment_id}\n")
            log.write(f"Parameters: {vars(params)}\n")
            log.write(f"Fold {fold} completed for selected landmarks {'_'.join(params.lmks)}.\n")
            log.write(f"Best validation loss: {best_val_loss}\n")   

    with open(log_file, "a") as log:
        log.write(f"Training completed for all folds.\n")

# # Testing or evaluation phase
if params.mode in ['test', 'eval']:

    # Training across folds
    if not params.iter_folds:
        iter_folds = range(params.n_split)
    else:
        iter_folds = params.iter_folds

    for fold in iter_folds:
        # Create a subdirectory for this fold within the experiment directory
        # EVALUATE # 
        # Initialize model, loss, optimizer
            # Create a subdirectory for this fold within the experiment directory
        fold_experiment_dir = os.path.join(experiment_directory, f'fold_{fold}')
        os.makedirs(fold_experiment_dir, exist_ok=True)
        model, criterion, optimizer, best_criteria = get_fresh_model(params)
        # Load the best model checkpoint
        model, optimizer, epoch, best_val_loss = load_checkpoint(model, optimizer, params.model_dir)
        print(f"Best validation loss was {best_val_loss}") 
        # Load test data
        test_dl = get_test_dl(params, fold, transformations=transformations)
        
        global_wandb_steps = {'train_loss': 0, 'val_loss': 0, 'epoch': 0}
        # Final evaluation on test set
        _, ep_scores, global_wandb_steps = infer_one_ep(
            model, test_dl, criterion, params.device, global_wandb_steps, 
            use_wandb=False, 
            extract_via=params.extract_via, 
            eval=True, 
            save_dir=fold_experiment_dir,
            radius_eval=params.radius_eval, 
            radius_num=params.radius_num,  
            lmks=params.lmks,
            )
        
        for lmk in params.lmks:
            print(f"Test d-mean Score for {lmk}: ", ep_scores[f"dmean_{lmk}"].mean(), " +/-", ep_scores[f"dmean_{lmk}"].std())

# if params.mode in ['test', 'eval']:
#     # Ensure supported execution mode
#     if params.training_mode != 'one-by-one':
#         raise ValueError(f"Unsupported execution mode: {params.execution}. "
#                          "Multiple landmarks cause high computation costs.")

#     # Loop through landmarks
#     for lmk in params.lmks:
#         if not params.iter_folds:
#             iter_folds = range(params.n_split)
#         else:
#             iter_folds = params.iter_folds

#         aela = []
#         for fold in iter_folds:
#             # Create a subdirectory for this fold within the experiment directory
#             fold_experiment_dir = os.path.join(experiment_directory, f'fold_{fold}')
#             os.makedirs(fold_experiment_dir, exist_ok=True)
#             # Initialize model, loss, optimizer
#             model, criterion, optimizer, best_criteria = get_fresh_model(params)
#             # Load the best model checkpoint
#             model_dir = f'runs/{params.model_dir}/best_fold{fold}.pt'
#             model, optimizer, epoch, best_val_loss = load_checkpoint(model, optimizer, model_dir)
#             print(f"Best validation loss was {best_val_loss}") 
#             # Load test data
#             test_dl = get_test_dl(params, fold, transformations=transformations)
#             # Final evaluation on test set
#             test_loss, ep_scores, ep_scores_curve, global_wandb_steps = infer_one_ep(
#                 model, test_dl, criterion, params.device, global_wandb_steps, 
#                 use_wandb=False, 
#                 extract_via=params.extract_via, 
#                 eval=True, 
#                 save_dir=fold_experiment_dir,
#                 radius_eval=params.radius_eval, 
#                 radius_num=params.radius_num,  
#                 lmk=lmk,
#                 )
            
#             print("Test d-mean Score: ", ep_scores['dmean'].mean(), " +/-", ep_scores['dmean'].std())
#             with open(log_file, "a") as log:
#                 log.write(f"Test d-mean Score:  {ep_scores['dmean'].mean()} +/- {ep_scores['dmean'].std()} \n")

#             ep_scores.to_csv(os.path.join(fold_experiment_dir, f'scores.csv'), index=False)
#             plot_aela_figure(
#                     torch.linspace(0, params.radius_eval, params.radius_num), ep_scores_curve.tolist(), 
#                     save_dir=os.path.join(fold_experiment_dir, f'curve.png')
#             )
#             # Save the ep_scores_curve as CSV, including lmk info in the filename
#             curve_csv_path = os.path.join(fold_experiment_dir, f'curve_{lmk}_{fold}.csv') 
#             np.savetxt(curve_csv_path, np.array(ep_scores_curve.tolist()), delimiter=",")

# python fetusnet.py test --lmks enR --num_fts 32 --loss emd --optim adam --lr 0.0001 --iter_folds 1 --model_dir archive/emd4enr_rn --prefix deneme 