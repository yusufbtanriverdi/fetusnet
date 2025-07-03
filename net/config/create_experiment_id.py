from datetime import datetime
import os

def create_experiment_id(params, create_directory=True):
    """    
    Generates a unique experiment ID using the current date and provided run name, appending a number if it already exists,
    and optionally creates a corresponding directory.

        params: An object with `prefix` and `lmks` attributes used as part of the experiment ID and directory name.
        create_directory (bool, optional): If True, creates a directory for the experiment under 'runs/'. Defaults to True.

        Returns:
            tuple:
                - experiment_directory (str or None): Path to the created experiment directory, or None if not created.
                - experiment_id (str): Unique experiment ID in the format 'YYYY-MM-DD/prefix[_n]'.

    Raises:
        OSError: If directory creation fails and `create_directory` is True.
    """

    current_time = datetime.now()
    date_str = current_time.strftime("%Y-%m-%d")
    base_experiment_id = f"{date_str}/{params.prefix}"
    experiment_directory = None
    counter = 1

    if create_directory:
        experiment_directory = f"runs/{base_experiment_id}"

        # Increment suffix until a unique directory name is found
        while os.path.exists(experiment_directory):
            experiment_id = f"{base_experiment_id}_{counter}"
            experiment_directory = f"runs/{experiment_id}"
            counter += 1

        os.makedirs(experiment_directory, exist_ok=True)

    return experiment_directory, date_str
