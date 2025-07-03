from net.dataset.MyDataset import MyDataset
import torchio as tio

def get_train_val_dl(splitted_dataframe, params, transformations):
    """
    Create training and validation DataLoaders from a pre-split dataframe.

    This function uses the provided split dataframe to generate `MyDataset` instances
    for training and validation. It then wraps them with `torchio.SubjectsLoader`
    to return ready-to-use DataLoaders.

    Args:
        splitted_dataframe (pd.DataFrame): DataFrame containing the full dataset with 
            a 'set' column:
                - 0: training samples
                - 1: validation samples
        params (Namespace): Configuration object with the following attributes:
            - sys (str): Base system path.
            - root (str): Dataset root folder.
            - generate (bool): Whether to generate additional features or labels.
            - alpha (float): Parameter passed to dataset.
            - eps (float): Parameter passed to dataset.
            - lmks (Any): Landmark information used in data loading.
            - batch_size_train (int): Batch size for training DataLoader.
            - batch_size_val (int): Batch size for validation DataLoader.
            - num_workers (int): Number of subprocesses used for data loading.
        transformations (Callable): A set of data transformations to apply to each sample.

    Returns:
        tuple:
            - train_dl (tio.SubjectsLoader): DataLoader for training data.
            - val_dl (tio.SubjectsLoader): DataLoader for validation data.
    """
    # Create training and validation datasets
    train_ds = MyDataset(
        splitted_dataframe[splitted_dataframe['set'] == 0].reset_index(drop=True), # Subset for training
        params.sys + params.root,                     # Root directory
        params.generate,                              # Data generation flag
        (params.g_alpha, params.g_eps),                   # Additional parameters
        params.lmks,                                          # Landmark information
        transformations,                              # Preprocessing transformations

    )
    val_ds = MyDataset(
        splitted_dataframe[splitted_dataframe['set'] == 1].reset_index(drop=True),   # Subset for validation
        params.sys + params.root,                     # Root directory
        params.generate,                              # Data generation flag
        (params.g_alpha, params.g_eps),                   # Additional parameters
        params.lmks,                                          # Landmark information
        transformations,                              # Preprocessing transformations
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


def get_test_dl(splitted_dataframe, params, transformations):
    """
    Create a DataLoader for the test set using the provided parameters and transformations.

    This function filters the input dataframe for test entries (where 'set' == 2),
    initializes a dataset, and wraps it in a PyTorch-Ignite DataLoader for batch inference.

    Args:
        splitted_dataframe (pd.DataFrame): The full dataframe with a 'set' column where:
                                           0 = train, 1 = val, 2 = test.
        params (Namespace or dict-like): Configuration object containing:
            - sys (str): System root path prefix.
            - root (str): Relative dataset root path.
            - generate (bool): Whether to use synthetic/generated data.
            - alpha (float): Hyperparameter passed to the dataset.
            - eps (float): Another dataset hyperparameter.
            - lmks (list): List of landmark names.
            - batch_size_test (int): Batch size for the test DataLoader.
            - num_workers (int): Number of worker threads for data loading.
        transformations (Callable): Transformations to apply to each sample (e.g., augmentation, normalization).

    Returns:
        torch.utils.data.DataLoader: DataLoader instance for the test set, ready for evaluation/inference.
    """

    # Create test dataset
    test_ds = MyDataset(
        splitted_dataframe[splitted_dataframe['set'] == 2].reset_index(drop=True), # Subset for test
        params.sys + params.root,                    # Root directory
        params.generate,                             # Data generation flag
        (params.g_alpha, params.g_eps),                  # Additional parameters
        params.lmks,                                         # Landmark information
        transformations,                             # Preprocessing transformations

    )

    # Create DataLoader for the test dataset
    test_dl = tio.SubjectsLoader(
        dataset=test_ds,
        batch_size=params.batch_size_test,  # Batch size for testing
        shuffle=False,                      # Do not shuffle test data
        num_workers=params.num_workers,      # Number of worker threads
    )

    return test_dl
