parameters_default = {
    'experiments_path': '/path/to/experiments',
    'sys': 'linux',
    'desired_size': 128,
    'desired_spacings': 2,
    'drate': 2,
    'root': '/data',
    'dataset': ['clinic1', 'clinic2'],
    'n_split': 4,
    'test_patients': [7, 20, 35],
    'save_dir': '/output',
    'generate': 'default',
    'target_idx': [0, 1, 2],
    'alpha': 3,
    'eps': 0.5,
    'run_name': 'experiment_1',
    'use_wandb': False,
    'GPU': '0',
    'num_threads': 4,
    'seed': 42,
    'rescale': True,
    'lmks': ['prn'],
    'batch_size_train': 4,
    'batch_size_val': 1,
    'batch_size_test': 1,
    'num_workers': 1,
    'backbone': 'resunet3d',
    'training_mode': 'train-one-by-one',
    'epochs': 100,
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'lr_momentum': 0.9,
    'reduction': 'mean',
    'criterion': 'mse',
    'criticise': 'default',
    'type_draw': 'heatmap',
    'num_images': 10,
    'idx': 0,
    'wandbpro': 'heatmap3d'
}

parameters_help = {
    'mode': "EXECUTION MODE",
    'train': 'train model',
    'test': 'test model',
    'train_test': 'train and test model',
    'explainability': 'explainability mode',
    'script_prepare': 'prepare dataset',
    'script_split': 'split dataset',
    'script_rotate': 'rotate / align 3D images',
    'script_generate_targets': 'generate targets',
    'help': 'show this message',

    'dataset_path': 'Path to the dataset directory.',
    'experiments_path': 'Path where experiment results will be saved.',
    'sys': 'Operating system type.',
    'desired_size': 'Target size of input images.',
    'desired_spacings': 'Target spacing for resampling.',
    'drate': 'Downsampling rate.',
    'root': 'Root directory of the dataset.',
    'dataset': 'List of clinics included in the dataset.',
    'n_split': 'Number of splits for cross-validation.',
    'test_patients': 'List of patient IDs reserved for testing.',
    'save_dir': 'Directory to save processed data.',
    'generate': 'Target mode for data generation.',
    'target_idx': 'Indices of target landmarks.',
    'alpha': 'Alpha parameter for loss calculation.',
    'eps': 'Gaussian threshold for peak detection.',
    'run_name': 'Name of the current experiment run.',
    'use_wandb': 'Flag to enable logging with Weights & Biases.',
    'GPU': 'GPU ID to use for training.',
    'num_threads': 'Number of CPU threads for data loading.',
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
    'criterion': 'Loss function to use.',
    'criticise': 'Evaluation mode.',
    'type_draw': 'Type of visualization output.',
    'num_images': 'Number of images to visualize.',
    'idx': 'Index of the sample to visualize.',
    'mode': 'Script mode',
    'wandbpro': 'Project name to log in wandb.'
}

parameters_choices = {
    'execution': ['train', 'test'],
    'optimizer': ['adam', 'sgd'],
    'criterion': ['mse', 'dce', 'kld', 'emd'],
    'reduction': ['mean', 'sum', 'none'],
    'type_draw': ['heatmap', 'overlay'],
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