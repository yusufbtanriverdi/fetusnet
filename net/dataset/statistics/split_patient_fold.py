import pandas as pd
from sklearn.model_selection import GroupKFold


def split(dataframe, params):
    """ 
    Splits a dataframe into train, test, and validation sets using StratifiedGroupKFold.
    
    Args:
        dataframe (pd.DataFrame): The dataframe containing 'pid' (group identifier) and 'week' (target for stratification).
    
    Returns:
        tuple: Indices for train, test, and validation sets.
    """
    splitter = GroupKFold(n_splits=params.n_split)  # Added random_state for reproducibility
    # Split into train and test sets
    for num, fold in enumerate(splitter.split(dataframe, dataframe['week'], dataframe['pid'])): 
        train_ind, val_ind = fold
        # Map train_ind and val_ind back to original indices
        train_ind = dataframe.iloc[train_ind].index.values
        val_ind = dataframe.iloc[val_ind].index.values
        dataframe.loc[val_ind, 'fold'] = num
        print('Fold ', str(num), ' :', len(train_ind), '\n' , len(val_ind))
    
    return dataframe


def get_patient_numbers_per_fold(dataframe):
    return len(dataframe[dataframe['set'] == 1]['pid'].unique())
