import datetime
import os
import pandas as pd
import torchio as tio
import warnings
import wandb
import torch
import random
import numpy as np
import yaml

# from net.parameters.parameters import parameters_parsing
from net.config.parser import setup_config
from net.dataset.statistics.split import perform_crossfold_split
from net.dataset.statistics.prepare import perform_prepare
from net.dataset.preprocess import perform_preprocessing
from net.dataset.generate_targets import perform_generate
from logger_setup import setup_logger
from net.dataset.utility.loaders import get_train_val_dl, get_test_dl
from net.model.utility.get_fresh_model import get_fresh_model
from net.phases.train import train_one_ep
from net.phases.infer import infer_one_ep
from net.model.utility.checkpoints import load_checkpoint, save_checkpoint
from net.config.wandb import initialize_wandb
from net.plot.volumes import perform_plot_3d, start_game_3d
from scripts import script_concept_fig
from utils import *

def pipe(dataframe, experiment_dir, params, transformations, global_wandb_steps, validate=True):
    train_dl, val_dl = get_train_val_dl(dataframe, params, transformations=transformations)
    if len(val_dl) == 0: val_dl = get_test_dl(dataframe, params, transformations=transformations)
    logger.info(f"Training - validation subset sizes: {len(train_dl.dataset), len(val_dl.dataset)}")
    # Initialize model, loss, optimizer
    model, criteria, optimizer, multi_noise_loss, best_val_loss = get_fresh_model(params)
    logger.info(optimizer)
    for criterion in criteria:
        logger.info(criterion)

    train_losses = []
    val_losses = []
    val_loss = torch.inf
    # Epoch training loop
    for epoch in range(0, params.epochs):
        logger.info(f" \n[Epoch {epoch + 1}/{params.epochs}] \n ")

        # Train for one epoch
        train_loss, global_wandb_steps = train_one_ep(
            model, 
            train_dl, 
            criteria, 
            multi_noise_loss if params.mnl else None,
            optimizer, 
            params.device, 
            global_wandb_steps,
            params.use_wandb,
            progress_bar=params.progress_bar
        )

        if validate:
            val_loss, global_wandb_steps, ep_scores = infer_one_ep(
                model, 
                val_dl, 
                criteria, 
                multi_noise_loss if params.mnl else None,
                params.device, 
                global_wandb_steps, 
                params.use_wandb,
                params.detector, 
                params.progress_bar,     
                params.lmks, 
                experiment_dir,
                check_visibility=params.check_visibility,
                eval=False, 
            )
        
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, val_loss, 
                                os.path.join(experiment_dir, f'best.pt'))

        # Save last model
        save_checkpoint(model, optimizer, epoch, val_loss, 
                        os.path.join(experiment_dir, f'last.pt'))

        logger.info(f"Train loss: {train_loss}")
        logger.info(f"Validation loss: {val_loss}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        global_wandb_steps = log_epoch_to_wandb(train_loss, val_loss, ep_scores, params, global_wandb_steps)

    return train_losses, val_losses

print("Torch cuda is available? ", torch.cuda.is_available())
# Suppress UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
# Initialize parameters
# params = parameters_parsing()
params = setup_config()
# Apply reproducibility settings
# Set random seed from parameters for reproducibility
torch_seed = params.reproducibility_.model_seed
# Set environment variables for deterministic behavior
os.environ["PYTHONHASHSEED"] = str(torch_seed)  # Python hash randomization
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # CUDA determinism configuration
# Set PyTorch seeds for reproducibility across CPU and GPU
torch.manual_seed(torch_seed)  # CPU seed
torch.cuda.manual_seed(torch_seed)  # GPU seed (current device)
torch.cuda.manual_seed_all(torch_seed)  # All GPU devices
# Set NumPy and random module seeds
np.random.seed(torch_seed)  # NumPy random seed
random.seed(torch_seed)  # Python random module seed
# Enable deterministic algorithms in PyTorch
torch.backends.cudnn.deterministic = params.reproducibility_.deterministic  # Use deterministic CUDA algorithms
torch.backends.cudnn.benchmark = params.reproducibility_.benchmark  # Disable auto-tuning (slower but reproducible)
# torch.use_deterministic_algorithms(True)  # Force deterministic behavior globally

# Convert to dictionary (if Namespace or similar)
params_as_dict = namespace_to_dict(params)  # or: params.__dict__ if vars() doesn't work
if params.mode in ['test', 'interactive_plot', 'interactive_game'] or params.resume:
    experiment_dir = params.checkpoint_
else:
    experiment_dir, experiment_name = create_experiment_id(params.prefix)
    # Write to JSON
    with open(os.path.join(experiment_dir, 'params.yaml'), 'w') as f:
        yaml.dump(params_as_dict, f, indent=6)

stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# save_experiment_parameters(experiment_dir, experiment_name, params, stamp)

# Initialize logger
logger = setup_logger(experiment_dir)
logger.info(f"Experiment directory created at: {experiment_dir}")
logger.info(f"Experiment started and parameters are saved. This is the experiment dir: {experiment_dir} ")
logger.info(f"System path set to: {params.dataset_.sys}")
logger.info(f"Device set to: {params.device}")
logger.info("Searching for master dataframe in system+dataset_root... \n If you did not prepare such dataframe or you think it misses some samples, please re-run this script in prepare mode and specify datasets")

# TODO: Test this mode.
if params.mode == 'prepare':
    main_dataframe_path = perform_prepare(params) # Create info frames

# Load or create sinfo dataframe
main_dataframe_path = os.path.join(params.dataset_.sys, params.dataset_.root, params.dataset_.dataframe + '.csv')
if os.path.exists(main_dataframe_path):
    main_dataframe = pd.read_csv(main_dataframe_path)
else:
    logger.error(f"Master dataframe not found @ {main_dataframe_path}.")
    raise FileNotFoundError("Master dataframe not found.")

if params.mode == 'presplit':
    split_dataframe_path = perform_crossfold_split(main_dataframe, params)

# TODO: Test rotate, no rotate.
# TODO: Build pipe for standardization.
# Rotate/alignment step
if params.mode == 'preprocess':
    perform_preprocessing(main_dataframe, params, logger)

logger.info(f"Master dataframe path: {main_dataframe_path}")
logger.info(f"Total number of input in master dataframe: {len(main_dataframe)}")
logger.info(f"Training - validation - test subset sizes: {len(main_dataframe[main_dataframe['set'] == 0]), len(main_dataframe[main_dataframe['set'] == 1]), len(main_dataframe[main_dataframe['set'] == 2])}")

# Define transformations for 3D images
transforms = [
                tio.RescaleIntensity((0, 1)), 
                # tio.Flip(axes=('L',)),
              ] if params.train_.rescale_inputs else []
transformations = tio.Compose(transforms)
logger.info("Transformations are created.")
exit()

if params.mode == 'test_loaders':
    logger.info(f"I have been asked to test loaders... Here it goes.")
    train_dl, val_dl = get_train_val_dl(main_dataframe, params, transformations=transformations)
    test_loaders(train_dl)
    test_loaders(val_dl)

logger.info(f"I am cleaning the dataframe from non-existing files and nonfrontal cases... {len(main_dataframe)}")
main_dataframe = update_dataframe(main_dataframe, params)
logger.info(f"After cleaning, total number of input in master dataframe: {len(main_dataframe)}")
logger.info(f"!__[U]__! Training - validation - test subset sizes: {len(main_dataframe[main_dataframe['set'] == 0]), len(main_dataframe[main_dataframe['set'] == 1]), len(main_dataframe[main_dataframe['set'] == 2])}")

if params.mode == 'generate':
    main_dataframe = main_dataframe.loc[main_dataframe['npid'].isin(params.test_patients)]
    perform_generate(main_dataframe, experiment_dir, params)

if params.mode == 'plot_3d':
    main_dataframe = main_dataframe.loc[main_dataframe['npid'].isin(params.test_patients)]
    if len(main_dataframe) == 0:
        raise KeyError("No test patient is given. Aborting mission...")
    perform_plot_3d(main_dataframe, experiment_dir, params)

if params.mode == 'game_3d':
    main_dataframe = main_dataframe.loc[main_dataframe['npid'].isin(params.test_patients)]
    if len(main_dataframe) == 0:
        raise KeyError("No test patient is given. Aborting mission...")
    start_game_3d(main_dataframe, experiment_dir, params)

if params.mode == 'script_concept':
    logger.info("Running script_concept_fig.py")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # test_patients = getattr(params, 'test_patients', [''])
    print(params.test_patients)
    main_dataframe.loc[main_dataframe['npid'].isin(params.test_patients), "set"] = 2
    # Filter test patients first
    test_dl = get_test_dl(main_dataframe.loc[main_dataframe['npid'].isin(params.test_patients)], params, transformations=transformations)
    logger.info(f"!__[U]__! Test subset size: {len(test_dl)}")
    if len(test_dl) == 0: 
        test_dl = get_test_dl(main_dataframe, params, transformations)
        logger.info(f"!__[U]__! Couldn't find test patients, switch to entire test subset. Subset size: {len(test_dl)}")

        if len(test_dl) == 0:         
                _, test_dl = get_train_val_dl(main_dataframe, params, transformations=transformations)
    script_concept_fig.create_disc_figure(test_dl, params.loss_params['w'], save_dir='figures/')

global_wandb_steps = {'train_loss': 0, 
                      'val_loss': 0, 
                      'epoch': 0}

if params.mode == 'train':
    logger.info("I am starting to train")
    if params.use_wandb: initialize_wandb(params, experiment_name+'_'+params.split)
    logger.info(f"\n--- Training {params.split} --- \n")
    # Initialize model, loss, optimizer
    model, criteria, optimizer, multi_noise_loss, _ = get_fresh_model(params)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    epoch = 0
    if params.resume:
        experiment_dir = params.checkpoint_dir
        model_dir = experiment_dir + '/' + params.use_model + '.pt'
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
        model, optimizer, epoch, best_val_loss = load_checkpoint(model, optimizer, model_dir) 
        logger.info(f"Resuming training from epoch {epoch} with best validation loss {best_val_loss}.")
        logger.info(f"Resuming training from checkpoint loaded at {stamp}.")
    
    train_losses, val_losses = pipe(main_dataframe, experiment_dir, params, transformations, global_wandb_steps)
    # Combine losses into a DataFrame
    loss_df = pd.DataFrame({
        'epoch': epoch + np.arange(len(train_losses)),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    # Save to CSV or any preferred format
    loss_df.to_csv(os.path.join(experiment_dir, f"losses_{stamp}.csv"), index=False)

    model_dir = experiment_dir + '/' + params.use_model + '.pt'
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
    
    model, optimizer, epoch, best_val_loss = load_checkpoint(model, optimizer, model_dir) 
    logger.info(f"\n--- Testing {params.use_model} model from {model_dir} --- \n")
        
    test_dl = get_test_dl(main_dataframe, params, transformations=transformations)
    if len(test_dl) == 0: _, test_dl = get_train_val_dl(main_dataframe, params, transformations=transformations)
    test_loss, global_wandb_steps, test_scores  = infer_one_ep(
            model, 
            test_dl, 
            criteria, 
            multi_noise_loss if params.mnl else None,
            params.device, 
            global_wandb_steps, 
            params.use_wandb, 
            params.detector, 
            params.progress_bar,     
            params.lmks, 
            experiment_dir,            
            check_visibility=params.check_visibility,
            eval=True,
            radius_eval=params.radius_eval,
            radius_num=params.radius_num,
            save_targets=params.save_targets, 
            save_outputs=params.save_outputs,
            show_figures=True,
            loss_params=params.loss_params
            )
    
    logger.info(f"Test loss: {test_loss}")
    logger.info(f"Test scores: \n {test_scores.drop(['nsid'], axis=1).mean()}")
    counter = 1
    save_scores = os.path.join(experiment_dir, "test_scores.csv")
    while os.path.exists(save_scores):
        save_scores = os.path.join(experiment_dir, f"test_scores_{counter}.csv")
        counter += 1
    test_scores.to_csv(save_scores, index=False)

    counter = 1
    save_mean_scores =os.path.join(experiment_dir, "test_scores_mean.csv")
    while os.path.exists(save_mean_scores):
        save_mean_scores = os.path.join(experiment_dir, f"test_scores_mean_{counter}.csv")
        counter += 1
    test_scores.drop(['nsid'], axis=1).mean().to_csv(save_mean_scores)

    if params.use_wandb: wandb.finish()

if params.mode == 'test':
    params.use_wandb = False
    logger.info("I am starting to test")
    # if params.use_wandb: initialize_wandb(params, experiment_name+'_'+params.split)
    logger.info(f"\n--- Testing {params.split} --- \n")
    # Initialize model, loss, optimizer
    model, criteria, optimizer, multi_noise_loss, _ = get_fresh_model(params)
    # train_losses, val_losses = pipe(main_dataframe, experiment_dir, params, transformations, global_wandb_steps)

    experiment_dir = params.checkpoint_dir
    model_dir = experiment_dir + '/' + params.use_model + '.pt'
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
    
    model, optimizer, epoch, best_val_loss = load_checkpoint(model, optimizer, model_dir) 
    logger.info(f"\n--- Testing {params.use_model} model from {model_dir} --- \n")
    
    # test_patients = getattr(params, 'test_patients', [''])
    print(params.test_patients)
    main_dataframe.loc[main_dataframe['npid'].isin(params.test_patients), "set"] = 2
    # Filter test patients first
    test_dl = get_test_dl(main_dataframe.loc[main_dataframe['npid'].isin(params.test_patients)], params, transformations=transformations)
    logger.info(f"!__[U]__! Test subset size: {len(test_dl)}")
    if len(test_dl) == 0: 
        test_dl = get_test_dl(main_dataframe, params, transformations)
        logger.info(f"!__[U]__! Couldn't find test patients, switch to entire test subset. Subset size: {len(test_dl)}")

        if len(test_dl) == 0:         
                _, test_dl = get_train_val_dl(main_dataframe, params, transformations=transformations)
                logger.info(f"!__[U]__! Switched to val subset. Subset size: {len(test_dl)}")

    test_loss, global_wandb_steps, test_scores  = infer_one_ep(
            model, 
            test_dl, 
            criteria, 
            multi_noise_loss if params.mnl else None,
            params.device, 
            global_wandb_steps, 
            params.use_wandb, 
            params.detector, 
            params.progress_bar,     
            params.lmks, 
            experiment_dir,            
            check_visibility=params.check_visibility,
            eval=True,
            radius_eval=params.radius_eval,
            radius_num=params.radius_num,
            save_targets=params.save_targets, 
            save_outputs=params.save_outputs,
            show_figures=True,
            loss_params=params.loss_params
            )
    
    logger.info(f"Test loss: {test_loss}")
    logger.info(f"Test scores: \n {test_scores.drop(['nsid'], axis=1).mean()}")
    counter = 1
    save_scores = os.path.join(experiment_dir, "test_scores.csv")
    while os.path.exists(save_scores):
        save_scores = os.path.join(experiment_dir, f"test_scores_{counter}.csv")
        counter += 1
    test_scores.to_csv(save_scores, index=False)

    counter = 1
    save_mean_scores =os.path.join(experiment_dir, "test_scores_mean.csv")
    while os.path.exists(save_mean_scores):
        save_mean_scores = os.path.join(experiment_dir, f"test_scores_mean_{counter}.csv")
        counter += 1
    test_scores.drop(['nsid'], axis=1).mean().to_csv(save_mean_scores)
