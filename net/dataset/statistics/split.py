import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split

def perform_split(master_dataframe, params):
    """
    Perform data splitting and annotate master_dataframe with 'set' and optionally 'fold'.
    0 = train, 1 = val, 2 = test
    """
    df = master_dataframe.copy()
    df['set'] = -1  # Initialize all to unassigned
    df['fold'] = -1  # Only meaningful for crossfold

    split_mode = params.split
    test_patients = set(params.test_patients) if hasattr(params, 'test_patients') else set()
    n_splits = getattr(params, 'n_split', 5)
    random_state = getattr(params, 'random_state', 42)
    datasets = getattr(params, 'dataset', [])

    if split_mode == 'crossfold':
        # Assign test patients first
        if test_patients:
            df.loc[df['pid'].isin(test_patients), 'set'] = 2
        remaining_df = df[df['set'] == -1]

        unique_pids = remaining_df['pid'].unique()
        kf = GroupKFold(n_splits=n_splits)

        fold_dfs = []
        for fold_idx, (train_idx, val_idx) in enumerate(
            kf.split(unique_pids, groups=unique_pids)
        ):
            train_pids = unique_pids[train_idx]
            val_pids = unique_pids[val_idx]

            fold_train = remaining_df[remaining_df['pid'].isin(train_pids)].copy()
            fold_val = remaining_df[remaining_df['pid'].isin(val_pids)].copy()

            fold_train['set'] = 0
            fold_val['set'] = 1
            fold_train['fold'] = fold_idx
            fold_val['fold'] = fold_idx

            fold_dfs.append(fold_train)
            fold_dfs.append(fold_val)

        folds_df = pd.concat(fold_dfs, ignore_index=True)

        # Test patients already present in df (with set=2)
        final_df = pd.concat([
            folds_df,
            df[df['set'] == 2]
        ], ignore_index=True).sort_values(by='pid').reset_index(drop=True)

        return final_df

    elif split_mode == 'splitthecake':
        # Assign test set first
        if test_patients:
            df.loc[df['pid'].isin(test_patients), 'set'] = 2
            remaining_df = df[df['set'] == -1]
        else:
            remaining_df = df.copy()

        unique_pids = remaining_df['pid'].unique()
        train_pids, val_pids = train_test_split(
            unique_pids, test_size=0.2, random_state=random_state, shuffle=True
        )

        df.loc[df['pid'].isin(train_pids), 'set'] = 0
        df.loc[df['pid'].isin(val_pids), 'set'] = 1
        return df

    elif split_mode == 'splitthecakeacross':
        assert len(datasets) == 2, "splitthecakeacross requires exactly two datasets"

        first_ds_mask = df['source'] == datasets[0]
        second_ds_mask = df['source'] == datasets[1]

        first_ds_df = df[first_ds_mask]
        # second_ds_df = df[second_ds_mask]

        unique_pids = first_ds_df['pid'].unique()
        train_pids, val_pids = train_test_split(
            unique_pids, test_size=0.2, random_state=random_state, shuffle=True
        )

        df.loc[first_ds_mask & df['pid'].isin(train_pids), 'set'] = 0
        df.loc[first_ds_mask & df['pid'].isin(val_pids), 'set'] = 1
        df.loc[second_ds_mask, 'set'] = 2  # test set = entire second dataset
        return df

    elif split_mode == 'getalienshere':
        # Load alien test externally and mark in a separate dataframe (won't fit into master_df directly)
        main_mask = ~df['pid'].isin(test_patients) if test_patients else pd.Series([True]*len(df))
        main_df = df[main_mask].copy()

        unique_pids = main_df['pid'].unique()
        train_pids, val_pids = train_test_split(
            unique_pids, test_size=0.2, random_state=random_state, shuffle=True
        )

        df.loc[df['pid'].isin(train_pids), 'set'] = 0
        df.loc[df['pid'].isin(val_pids), 'set'] = 1

        # Alien test data will be returned separately or handled outside
        alien_test_df = pd.read_csv(params.alien_test_path)
        alien_test_df['set'] = 2

        # Important: Make sure alien_test_df columns align with df
        # If any columns missing, add them with NaNs to match df columns
        for col in df.columns:
            if col not in alien_test_df.columns:
                alien_test_df[col] = pd.NA
        for col in alien_test_df.columns:
            if col not in df.columns:
                df[col] = pd.NA

        # Reorder alien_test_df columns to match df exactly
        alien_test_df = alien_test_df[df.columns]

        # Concatenate the alien test data with main df
        combined_df = pd.concat([df, alien_test_df], ignore_index=True)
        return combined_df

    else:
        raise ValueError(f"Unknown split mode: {split_mode}")
