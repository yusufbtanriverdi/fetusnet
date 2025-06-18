from datetime import datetime
import os

def create_experiment_id(params, create_directory=True):
    """    
    Generates a unique experiment ID using the current date and a provided run name, and optionally creates a corresponding directory.

        params: An object with a `run_name` attribute, used as part of the experiment ID and directory name.
        create_directory (bool, optional): If True, creates a directory for the experiment under 'runs/'. Defaults to True.

        tuple:
            - experiment_directory (str or None): The path to the created experiment directory, or None if not created.
            - experiment_id (str): The unique experiment ID in the format 'YYYY-MM-DD/run_name'.

    Raises:
        OSError: If directory creation fails and `create_directory` is True.

    Example:
        >>> class Params: run_name = "test_run"
        >>> create_experiment_id(Params())
        ('runs/2024-06-13/test_run', '2024-06-13/test_run')"""

    # Get the current timestamp
    current_time = datetime.now()

    # Generate a directory name for the current day and a unique experiment ID using the run name
    date_str = current_time.strftime("%Y-%m-%d")
    lmks_str = "_".join(str(lmk) for lmk in params.lmks)
    experiment_id = f"{date_str}/{params.prefix}_{lmks_str}"

    # Initialize the experiment directory variable
    experiment_directory = None

    # Create the experiment directory if requested
    if create_directory:
        experiment_directory = f'runs/{experiment_id}'
        os.makedirs(experiment_directory, exist_ok=True)

    # Return the experiment directory path and experiment ID
    return experiment_directory, experiment_id