import pandas as pd
from sklearn.model_selection import GroupKFold
import os


def main(dataframe, params):
    """ 
    Splits a dataframe into train, test, and validation sets using StratifiedGroupKFold.
    
    Args:
        dataframe (pd.DataFrame): The dataframe containing 'pid' (group identifier) and 'week' (target for stratification).
    
    Returns:
        tuple: Indices for train, test, and validation sets.
    """

    # test_patients = [7,
    #                 19,
    #                 30,
    #                 51
    #                 ]
    test_patients = params.test_patients

    dataframe = pd.DataFrame.from_records(dataframe)
    dataframe.loc[:, 'test'] = 0  # Initialize the 'test' column with zeros
    test_ind = dataframe['pid'].isin(test_patients)  # Vectorized approach
    dataframe.loc[test_ind, 'test'] = 1
    dataframe.to_csv(params.sys + params.root + 'sinfo.csv', index=False)


    dataframe = dataframe[dataframe['landmark_antonia_found']].reset_index(drop=True)
    dataframe = dataframe[dataframe['test'] != 1].reset_index(drop=True)

    splitter = GroupKFold(n_splits=params.n_split)  # Added random_state for reproducibility
    
    # Split into train and test sets
    for num, fold in enumerate(splitter.split(dataframe, dataframe['week'], dataframe['pid'])): 
        train_ind, val_ind = fold
        # Map train_ind and val_ind back to original indices
        train_ind = dataframe.iloc[train_ind].index.values
        val_ind = dataframe.iloc[val_ind].index.values
        dataframe.loc[train_ind, 'set'] = 0
        dataframe.loc[val_ind, 'set'] = 1

        print('Fold ', str(num), ' :', len(train_ind), '\n' , len(val_ind))
        dataframe.to_csv(params.sys + params.root + 'sinfo__fold' + str(num) + '__.csv', index=False)

    return dataframe