parameters_default = {
    'os': 'linux',
    'desired_size': 128,
    'desired_spacings': 2,
    'drate': 2,
    'root': '/Rotated/Processed/',
    'raw_dir': '',
    'dataset': ['Casos Mar'],
    'n_split': 4,
    'generate': 'gaussian',
    'target_idx': [0, 1, 2],
    'alpha': 3,
    'eps': 1e-6,
    'run_name': 'experiment_1',
    'use_wandb': False,
    'device': 'cuda',
    'seed': 42,
    'rescale': True,
    'lmks': ['prn'],
    'batch_size_train': 1,
    'batch_size_val': 1,
    'batch_size_test': 1,
    'num_workers': 1,
    'backbone': 'resunet3d',
    'training_mode': 'one-by-one',
    'epochs': 100,
    'optimizer': 'sgd',
    'learning_rate': 0.001,
    'lr_momentum': 0.9,
    'reduction': 'mean',
    'loss': ['mse'],
    'criticise': 'default',
    'type_draw': 'heatmap',
    'num_images': 10,
    'idx': 0,
    'wandbpro': 'heatmap3d',
    'model_dir': '',
    'num_fts': 64,
    'iter_folds': None,
    'resume': False,
    'exp_dir': '',
    'radii_eval': 40,
    'radii_num': 100,
}

parameters_help = {
    'mode': "EXECUTION MODE",
    'train': 'train model',
    'test': 'test model',
    'raw_dir': 'raw images directiory',
    'train_test': 'train and test model',
    'explainability': 'explainability mode',
    'script_prepare': 'prepare dataset',
    'script_split': 'split dataset',
    'script_rotate': 'rotate / align 3D images',
    'script_generate_targets': 'generate targets',
    'help': 'show this message',

    'dataset_path': 'Path to the dataset directory.',
    'os': 'Operating system type.',
    'desired_size': 'Target size of input images.',
    'desired_spacings': 'Target spacing for resampling.',
    'drate': 'Downsampling rate.',
    'root': 'Root directory of the dataset.',
    'dataset': 'List of clinics included in the dataset.',
    'n_split': 'Number of splits for cross-validation.',
    'test_patients': 'List of patient IDs reserved for testing.',
    'generate': 'Target mode for data generation.',
    'target_idx': 'Indices of target landmarks.',
    'alpha': 'Alpha parameter for Gaussian heatmaps.',
    'eps': 'Gaussian threshold for peak detection.',
    'run_name': 'Name of the current experiment run.',
    'use_wandb': 'Flag to enable logging with Weights & Biases.',
    'device': 'Device to run the model on (e.g., cuda, cpu).',
    'seed': 'Random seed for reproducibility.',
    'rescale': 'Boolean flag for image rescaling.',
    'lmks': 'List of landmarks to detect.',
    'batch_size_train': 'Batch size for training.',
    'batch_size_val': 'Batch size for validation.',
    'batch_size_test': 'Batch size for testing.',
    'num_workers': 'Number of worker threads for data loading.',
    'backbone': 'Backbone network architecture.',
    'training_mode': 'Training mode (train landmarks one-by-one).',
    'epochs': 'Total number of training epochs.',
    'optimizer': 'Optimizer choice.',
    'learning_rate': 'Learning rate for training.',
    'lr_momentum': 'Momentum for SGD optimizer.',
    'reduction': 'Reduction method for loss computation.',
    'loss': 'Loss function to use. Accepts a list of loss functions.',
    'criticise': 'Evaluation mode.',
    'type_draw': 'Type of visualization output.',
    'num_images': 'Number of images to visualize.',
    'idx': 'Index of the sample to visualize.',
    'mode': 'Script mode',
    'wandbpro': 'Project name to log in wandb.',
    'model_dir': 'Model directory for test.',
    'num_fts': 'Number of features for backbone.',
    'iter_folds': 'Cross-validation folds selected (optional).',
    'resume': 'Resume training from a checkpoint.',
    'exp_dir': 'Directory for saving experiment results.',
    'radii_eval': 'Radius evaluation mode.',
    'radii_num': 'Number of radii to evaluate.',
}

parameters_choices = {
    'execution': ['train', 'test', 'script_generate_targets'],
    'optimizer': ['adam', 'sgd'],
    'reduction': ['mean', 'sum', 'none'],
    'type_draw': ['heatmap', 'overlay'],
    'generate': ['gaussian', 'distance'],
}

def print_help():
    print("Usage: python script.py [mode] [options]\n")
    print("Available Modes:")
    print("  train        Train the model")
    print("  test         Test the model")
    print("  validate     Validate the model\n")

    print("Arguments:\n")
    for key, desc in parameters_help.items():
        default_value = parameters_default.get(key, "None")
        print(f"  --{key}: {desc} (Default: {default_value})")
    
    print("\nChoices:")
    for key, choices in parameters_choices.items():
        print(f"  {key}: {choices}")

if __name__ == "__main__":
    print_help()