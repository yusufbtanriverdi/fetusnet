from net.dataset.MyDataset import MyDataset
import torchio as tio
import pandas as pd
from net.dataset.statistics.create_info_frames import update

def get_train_val_dl(lmk, num, params,  transformations):
    """
    Create DataLoaders for training and validation sets.

    Args:
        sinfo (pd.DataFrame): DataFrame containing dataset information.
        params: Configuration object containing hyperparameters.
        transformations: Preprocessing transformations.

    Returns:
        tuple: (train_dl, val_dl) - DataLoaders for training and validation.
    """
    # Split dataset into train and validation based on the 'set' column
    sinfo = pd.read_csv(params.sys + params.root + 'sinfo__fold' + str(num) + '__.csv')
    sinfo = update(sinfo, params)
    train_idx = sinfo.index[sinfo['set'] == 0].tolist()
    val_idx = sinfo.index[sinfo['set'] == 1].tolist()

    # Create datasets
    train_ds = MyDataset(sinfo.iloc[train_idx].reset_index(drop=True), params.sys + params.root, params.generate, (params.alpha, params.eps), lmk, transformations)
    val_ds = MyDataset(sinfo.iloc[val_idx].reset_index(drop=True), params.sys + params.root, params.generate, (params.alpha, params.eps), lmk, transformations)

    # Create DataLoaders
    train_dl = tio.SubjectsLoader(
        dataset=train_ds, 
        batch_size=params.batch_size_train,  # Fixed typo (batch_Size → batch_size)
        shuffle=True,  # Training should generally be shuffled
        num_workers=params.num_workers
    )

    val_dl = tio.SubjectsLoader(
        dataset=val_ds, 
        batch_size=params.batch_size_val, 
        shuffle=False, 
        num_workers=params.num_workers
    )

    return train_dl, val_dl


def get_test_dl(sinfo, params, lmk, transformations):
    """
    Create a DataLoader for the test set.

    Args:
        sinfo (pd.DataFrame): DataFrame containing dataset information.
        params: Configuration object containing hyperparameters.
        transformations: Preprocessing transformations.

    Returns:
        DataLoader: Test DataLoader.
    """
    # Get test indices where 'set' column is 2
    test_idx = sinfo.index[sinfo['set'] == 2].tolist()

    # Create dataset and DataLoader
    test_ds = MyDataset(sinfo[test_idx], params.sys + params.root, params.generate, (params.alpha, params.eps), lmk, transformations)
    test_dl = tio.SubjectsLoader(
        dataset=test_ds, 
        batch_size=params.batch_size_test, 
        shuffle=False, 
        num_workers=params.num_workers
    )

    return test_dl
