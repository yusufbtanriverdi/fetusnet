import os
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

def perform_crossfold_split(master_dataframe, params):
    """
    Perform data splitting and write resulting dataframes to disk.
    Returns the path to the last written file (or last fold if crossfold).
    """
    df = master_dataframe.copy()
    df['set'] = -1  # Initialize all to unassigned

    test_patients = params.split_.test_patients
    # Assign test patients first
    if test_patients:
        df.loc[df['npid'].isin(test_patients), 'set'] = 2

    test_ds = params.split_.test_ds
    if test_ds:
        df.loc[df['ds'].isin(test_ds), 'set'] = 2

    remaining_df = df[df['set'] == -1]
    n_splits = params.split_.n_split
    output_dir = os.path.join(params.dataset_.sys, params.dataset_.root) + '/' + params.dataset_.dataframe

    unique_pids = remaining_df['npid'].unique()
    kf = GroupKFold(n_splits=n_splits)

    last_path = None
    for fold_idx, (train_idx, val_idx) in enumerate(
        kf.split(unique_pids, groups=unique_pids)
    ):
        train_pids = unique_pids[train_idx]
        val_pids = unique_pids[val_idx]

        fold_train = remaining_df[remaining_df['npid'].isin(train_pids)].copy()
        fold_val = remaining_df[remaining_df['npid'].isin(val_pids)].copy()

        fold_train['set'] = 0
        fold_val['set'] = 1

        fold_df = pd.concat([fold_train, fold_val], ignore_index=True)
        # Add test patients for this fold
        test_df = df[df['set'] == 2].copy()
        if not test_df.empty:
            fold_df = pd.concat([fold_df, test_df], ignore_index=True)

        fold_df = fold_df.sort_values(by='npid').reset_index(drop=True)
        fold_path = output_dir + f'_fold{fold_idx}.csv'
        fold_df.to_csv(fold_path, index=False)
        last_path = fold_path

    return last_path