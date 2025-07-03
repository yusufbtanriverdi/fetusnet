# fetusnet
Facial landmark detection in fetal 3d ultrasound images @ University of Pompeu Fabra DTIC.

### Help with splitting
Parameter: split
Value	Description
"crossfold"	Perform cross-validation folds over the entire dataset(s), reserving any specified test patients as a separate test set.
"splitthecake"	Perform a single 80-20 train-validation split on the concatenated dataset(s), reserving any specified test patients as a separate test set.
"splitthecakeacross"	Perform an 80-20 train-validation split on the first dataset, then use the second dataset entirely as a test set.
"getalienshere"	Perform train/validation split on the main dataset(s) and load an external test dataset from a user-provided path for testing.


Detailed Behaviors


1. crossfold

    Combine all data sources into one large dataframe.

    If test_patients is non-empty:

        Remove test_patients from the data before splitting.

        Assign these patients exclusively to the test set.

    Perform K-Fold cross-validation (n_split folds) on the remaining data.

    For each fold:

        Use the fold’s validation split.

        Train on all other folds.

        Testing uses the reserved test_patients if provided; otherwise, use the validation fold or separate test fold if available.

    Useful for comprehensive cross-validation and robustness testing.

2. splitthecake

    Combine all data sources into one large dataframe.

    If test_patients is non-empty:

        Remove test_patients from the data before splitting.

        Assign these patients exclusively to the test set.

    Perform a single random split:

        80% train

        20% validation

    Testing uses the reserved test_patients if provided; otherwise, no test set or use validation as test.

    Useful for quick train/val splits on combined data.

3. splitthecakeacross

    Requires exactly two datasets provided in the dataset parameter.

    Use the first dataset for training and validation:

        Perform an 80-20 train-validation split on this dataset.

    Use the second dataset entirely as the test set.

    Ignore test_patients parameter in this mode.

    Useful for cross-dataset generalization tests.

4. getalienshere

    The main dataset(s) are concatenated and split into train and validation sets (default 80-20 split).

    The test set is NOT drawn from the main dataset(s).

    Instead, the test set is loaded from an external file (e.g., a CSV or DataFrame file) specified by a new parameter, say, alien_test_path.

    test_patients parameter is ignored in this mode.

    Useful for testing generalization on a completely external dataset not part of training.