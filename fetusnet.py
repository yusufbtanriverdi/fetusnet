import os
import pandas as pd
import torchio as tio

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
import warnings
import wandb
# Ignore all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize parameters
params = parameters_parsing()
# Create experiment directory & log
experiment_directory, experiment_id = create_experiment_id(params)
# Initialize WandB if enabled
if params.use_wandb:
    initialize_wandb(params)

# Initialize global tracking dictionary
global_wandb_steps = {'train_loss': 0, 'val_loss': 0, 'epoch': 0}

# Create local experiment log (timestamp, params)
log_file = os.path.join(experiment_directory, "experiment_log.txt")
with open(log_file, "w") as log:
    log.write(f"Experiment ID: {experiment_id}\n")
    log.write(f"Parameters: {vars(params)}\n")


if params.os in ['linux', 'l']:
    params.sys = "/media/yusuf/HDD 4TB/"
else:
    params.sys = "D:/"

if params.GPU != -1:
    params.device = 'cuda'
else:
    params.device = 'cpu'

if params.mode in ['script_prepare', 'script_split']:
    print(f" WELCOME TO {params.mode} MODE. \n .......................creating info frames......................................... \n")
    sinfo_df = create_info_frames.main(params)
if params.mode == 'script_split':
    print(f" WELCOME TO {params.mode} MODE. \n .......................splitting patients into train-test sets......................................... \n")
    sinfo_df = split_patient_fold.main(sinfo_df, params)

# Load or create sinfo dataframe
sinfo_path = params.sys + params.root + 'sinfo.csv'
if os.path.exists(sinfo_path):
    sinfo_df = pd.read_csv(sinfo_path) 
else: 
    raise FileNotFoundError
sinfo_df = sinfo_df[sinfo_df['landmark_antonia_found']].reset_index(drop=True)
print(len(sinfo_df))
sinfo_df = create_info_frames.update(sinfo_df, params)
print(len(sinfo_df))

# # 🔹 (TODO) Rotate/alignment step - Verify ground truth consistency
# if params.mode == 'script_rotate':
#     rotate.main(sinfo_df, params)


# 🔹 (TODO) Test - Generate target datasets
if params.mode == 'script_generate_targets':
    print(params.sys)
    generate_targets.main(sinfo_df, experiment_directory, params)

# Define transformations for 3D images
transforms = [tio.RescaleIntensity((0, 1))] if params.rescale else [] # In this version, only for input image.
transformations = tio.Compose(transforms)
# Training Phase
if params.mode in ['train', 'train_test']:
    # Check execution type
    if params.training_mode != 'one-by-one':
        raise ValueError(f"Unsupported execution mode: {params.training_mode}. "
                         "Multiple landmarks cause high computation costs.")

    for lmk in params.lmks:
        # Training across Folds
        for fold in range(params.n_split):
            print(f"\n--- Training Fold {fold + 1}/{params.n_split} ---")
            # Load Data
            train_dl, val_dl = get_train_val_dl(lmk, fold, params, transformations=transformations)

            # Initialize Model, Loss, Optimizer
            model, criterion, optimizer, best_criteria = get_fresh_model(params)

            # Epoch Training Loop
            for epoch in range(params.epochs):
                print(f"\n[Epoch {epoch + 1}/{params.epochs}]")

                train_loss, global_wandb_steps = train_one_ep(
                    model, train_dl, criterion, optimizer, params.device, 
                    wandb_steps=global_wandb_steps
                )

                val_loss, ep_scores, global_wandb_steps = infer_one_ep(
                    model, val_dl, criterion, params.device, 
                    wandb_steps=global_wandb_steps
                )

                wandb.log({'epoc/val_loss': val_loss, 'epoc/epoch': global_wandb_steps['epoch']})
                wandb.log({'epoc/train_loss': train_loss, 'epoc/epoch': global_wandb_steps['epoch']})
                
                # Save Best Model
                if val_loss < best_criteria:
                    best_criteria = val_loss
                    save_checkpoint(model, optimizer, epoch, val_loss, 
                                    os.path.join(experiment_directory, 'best.pt'))

                # Save Last Model
                save_checkpoint(model, optimizer, epoch, val_loss, 
                                os.path.join(experiment_directory, 'last.pt'))

            # Final Evaluation on Validation Set
            val_loss, ep_scores, global_wandb_steps = infer_one_ep(model, val_dl, criterion, params.device, 
                        wandb_steps=global_wandb_steps, eval=True)
            wandb.finish()
            # Re-initialize global tracking dictionary
            global_wandb_steps = {'train_loss': 0, 'val_loss': 0, 'epoch': 0}
            initialize_wandb(params, fold=fold+1)

if params.mode in ['test', 'eval']:
    
    # Check execution type
    if params.training_mode != 'one-by-one':
        raise ValueError(f"Unsupported execution mode: {params.execution}. "
                         "Multiple landmarks cause high computation costs.")

    for lmk in params.lmks:
        # Initialize Model, Loss, Optimizer
        model, criterion, optimizer, best_criteria = get_fresh_model(params)
        model_dir = f'runs/{params.model_dir}/best.pt'
        model, optimizer, epoch, best_val_loss = load_checkpoint(model, optimizer, model_dir)
        # Load Data
        test_dl = get_test_dl(params, lmk, transformations=transformations)

        # Final Evaluation on Validation Set
        infer_one_ep(model, test_dl, criterion, params.device, 
                        wandb_steps=global_wandb_steps, eval=True, save_dir=experiment_directory)