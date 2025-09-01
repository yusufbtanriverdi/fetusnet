import os
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

def perform_split(master_dataframe, params):
    """
    Perform data splitting and write resulting dataframes to disk.
    Returns the path to the last written file (or last fold if crossfold).
    """
    df = master_dataframe.copy()
    df['set'] = -1  # Initialize all to unassigned

    split_mode = params.split
    test_patients = getattr(params, 'test_patients', list())
    if not test_patients: test_patients = []
    n_splits = getattr(params, 'n_split', 5)
    random_state = getattr(params, 'seed', 42)
    datasets = getattr(params, 'dataset', [])
    output_dir = os.path.join(params.sys, params.root) + '/' + params.master_df

    if split_mode == 'crossfold':
        # Assign test patients first
        if test_patients:
            df.loc[df['npid'].isin(test_patients), 'set'] = 2
        remaining_df = df[df['set'] == -1]

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

    elif split_mode == 'splitthecake':
        # Assign test set first
        if test_patients:
            df.loc[df['npid'].isin(test_patients), 'set'] = 2
            remaining_df = df[df['set'] == -1]
        else:
            remaining_df = df.copy()

        unique_pids = remaining_df['npid'].unique()
        train_pids, val_pids = train_test_split(
            unique_pids, test_size=0.2, random_state=random_state, shuffle=True
        )

        df.loc[df['npid'].isin(train_pids), 'set'] = 0
        df.loc[df['npid'].isin(val_pids), 'set'] = 1

        out_path = output_dir + '_splitthecake.csv'
        df.to_csv(out_path, index=False)
        return out_path

    elif split_mode == 'splitthecakeacross':
        assert len(datasets) == 2, "splitthecakeacross requires exactly two datasets"

        # Assign test patients first
        if test_patients:
            df.loc[df['npid'].isin(test_patients), 'set'] = 2

        # Assign training set: all from dataset 0 not in test
        train_mask = (df['ds'] == datasets[0]) & (~df['npid'].isin(test_patients))
        df.loc[train_mask, 'set'] = 0

        # Assign validation set: all from dataset 1 not in test
        val_mask = (df['ds'] == datasets[1]) & (~df['npid'].isin(test_patients))
        df.loc[val_mask, 'set'] = 1

        out_path = output_dir + '_splitthecakeacross.csv'
        df.to_csv(out_path, index=False)
        return out_path

    elif split_mode == 'getalienshere':
        main_mask = ~df['npid'].isin(test_patients) if test_patients else pd.Series([True]*len(df))
        main_df = df[main_mask].copy()

        unique_pids = main_df['npid'].unique()
        train_pids, val_pids = train_test_split(
            unique_pids, test_size=0.2, random_state=random_state, shuffle=True
        )

        df.loc[df['npid'].isin(train_pids), 'set'] = 0
        df.loc[df['npid'].isin(val_pids), 'set'] = 1

        alien_test_df = pd.read_csv(params.alien_test_path)
        alien_test_df['set'] = 2

        for col in df.columns:
            if col not in alien_test_df.columns:
                alien_test_df[col] = pd.NA
        for col in alien_test_df.columns:
            if col not in df.columns:
                df[col] = pd.NA

        alien_test_df = alien_test_df[df.columns]
        combined_df = pd.concat([df, alien_test_df], ignore_index=True)

        # Find next available filename if exists
        base_path = output_dir + '_getalienshere'
        out_path = base_path + '.csv'
        idx = 1
        while os.path.exists(out_path):
            out_path = f"{base_path}_{params.prefix}{idx}.csv"
            idx += 1
        combined_df.to_csv(out_path, index=False)
        return out_path

    else:
        raise ValueError(f"Unknown split mode: {split_mode}")
