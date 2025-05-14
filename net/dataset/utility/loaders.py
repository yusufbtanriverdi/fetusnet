from net.dataset.MyDataset import MyDataset
import torchio as tio
import pandas as pd
from net.dataset.statistics.create_info_frames import update

def get_train_val_dl(lmk, num, params, transformations):
    """
    Create DataLoaders for training and validation sets.

    Args:
        lmk: Landmark information used for dataset creation.
        num (int): Fold number for cross-validation.
        params: Configuration object containing hyperparameters.
        transformations: Preprocessing transformations to apply.

    Returns:
        tuple: (train_dl, val_dl) - DataLoaders for training and validation sets.
    """
    # Load dataset information from a CSV file
    sinfo_path = f"{params.sys}{params.root}sinfo__fold{num}__.csv"
    sinfo = pd.read_csv(sinfo_path)

    # Update dataset information (e.g., preprocessing or additional metadata)
    sinfo = update(sinfo, params)

    # Split dataset into training and validation sets based on the 'set' column
    train_idx = sinfo.index[sinfo['set'] == 0].tolist()  # Indices for training set
    val_idx = sinfo.index[sinfo['set'] == 1].tolist()    # Indices for validation set

    # Create training and validation datasets
    train_ds = MyDataset(
        sinfo.iloc[train_idx].reset_index(drop=True), # Subset for training
        params.sys + params.root,                     # Root directory
        params.generate,                              # Data generation flag
        (params.alpha, params.eps),                   # Additional parameters
        lmk,                                          # Landmark information
        transformations,                              # Preprocessing transformations
        params.dist_matrix,                           # Distance matrix flag

    )
    val_ds = MyDataset(
        sinfo.iloc[val_idx].reset_index(drop=True),   # Subset for validation
        params.sys + params.root,                     # Root directory
        params.generate,                              # Data generation flag
        (params.alpha, params.eps),                   # Additional parameters
        lmk,                                          # Landmark information
        transformations,                              # Preprocessing transformations
        params.dist_matrix,                           # Distance matrix flag
    )

    # Create DataLoaders for training and validation datasets
    train_dl = tio.SubjectsLoader(
        dataset=train_ds,
        batch_size=params.batch_size_train,  # Batch size for training
        shuffle=True,                        # Shuffle training data
        num_workers=params.num_workers      # Number of worker threads
    )
    val_dl = tio.SubjectsLoader(
        dataset=val_ds,
        batch_size=params.batch_size_val,   # Batch size for validation
        shuffle=False,                      # Do not shuffle validation data
        num_workers=params.num_workers      # Number of worker threads
    )

    return train_dl, val_dl


def get_test_dl(params, num, lmk, transformations):
    """
    Create a DataLoader for the test set.

    Args:
        params: Configuration object containing hyperparameters.
        lmk: Landmark information used for dataset creation.
        transformations: Preprocessing transformations to apply.

    Returns:
        DataLoader: Test DataLoader.
    """
    # Load dataset information from a CSV file for the test set
    sinfo_path = f"{params.sys}{params.root}sinfo__fold{num}__.csv"
    sinfo = pd.read_csv(sinfo_path)

    # Update dataset information (e.g., preprocessing or additional metadata)
    sinfo = update(sinfo, params)
    
    if params.test_patients is None:
        print("Test patients list is empty. I will iterate all validation patients in current fold.")
        test_idx = sinfo.index[sinfo['set'] == 1].tolist()    # Indices for validation set

    else: 
        print(f"Test patients: {params.test_patients}")
        # Identify indices for test set based on patient IDs
        test_idx = sinfo.index[sinfo['pid'].isin(params.test_patients)].tolist()

    # Create test dataset
    test_ds = MyDataset(
        sinfo.iloc[test_idx].reset_index(drop=True), # Subset for test
        params.sys + params.root,                    # Root directory
        params.generate,                             # Data generation flag
        (params.alpha, params.eps),                  # Additional parameters
        lmk,                                         # Landmark information
        transformations,                             # Preprocessing transformations
        params.dist_matrix,                          # Distance matrix flag

    )

    # Create DataLoader for the test dataset
    test_dl = tio.SubjectsLoader(
        dataset=test_ds,
        batch_size=params.batch_size_test,  # Batch size for testing
        shuffle=False,                      # Do not shuffle test data
        num_workers=params.num_workers      # Number of worker threads
    )

    return test_dl
