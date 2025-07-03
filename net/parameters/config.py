parameters_default = {
    'os': 'linux',
    'desired_size': [128, 128, 128],
    'desired_spacings': [1.0, 1.0, 1.0],
    'root': '/Rotated/Processed/',
    'raw_dir': '',
    'dataset': ['Casos Mar'],
    'n_split': 4,
    'generate': 'gaussian',
    'target_idx': [0, 1, 2],
    'test_patients': [],
    'alpha': 3,
    'eps': 1e-6,
    'prefix': 'experiment_1',
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
    'epochs': 100,
    'optimizer': 'adam',
    'learning_rate': 0.0001,
    'lr_momentum': 0.9,
    'reduction': 'mean',
    'loss': ['mse'],
    'criticise': 'default',
    'wandbpro': 'fetusnetv1',
    'model_dir': '',
    'num_fts': 32,
    'iter_folds': [1],
    'resume': False,
    'exp_dir': '',
    'radius_eval': 40,
    'radius_num': 40,
    'extract_via': 'argmax',
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
    'root': 'Root directory of the dataset.',
    'dataset': 'List of clinics included in the dataset.',
    'n_split': 'Number of splits for cross-validation.',
    'test_patients': 'List of patient IDs reserved for testing.',
    'generate': 'Target mode for data generation.',
    'target_idx': 'Indices of target landmarks.',
    'alpha': 'Alpha parameter for Gaussian heatmaps.',
    'eps': 'Gaussian threshold for peak detection.',
    'prefix': 'Name of the current experiment run.',
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
    'epochs': 'Total number of training epochs.',
    'optimizer': 'Optimizer choice.',
    'learning_rate': 'Learning rate for training.',
    'lr_momentum': 'Momentum for SGD optimizer.',
    'reduction': 'Reduction method for loss computation.',
    'loss': 'Loss function to use. Accepts a list of loss functions.',
    'criticise': 'Evaluation mode.',
    'mode': 'Script mode',
    'wandbpro': 'Project name to log in wandb.',
    'model_dir': 'Model directory for test.',
    'num_fts': 'Number of features for backbone.',
    'iter_folds': 'Cross-validation folds selected (optional).',
    'resume': 'Resume training from a checkpoint.',
    'exp_dir': 'Directory for saving experiment results.',
    'radius_eval': 'Radius evaluation mode.',
    'radius_num': 'Number of radii to evaluate.',
    'extract_via': 'Method to extract peak location from heatmaps.',
}

parameters_choices = {
    'execution': ['train', 'test', 'script_generate_targets'],
    'optimizer': ['adam', 'sgd'],
    'reduction': ['mean', 'sum', 'none'],
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