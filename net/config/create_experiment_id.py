from datetime import datetime
import os

def create_experiment_id(params, create_directory=True):
    """
    Creates a unique experiment ID based on the current timestamp and a given run name.
    Optionally creates a directory for the experiment.

    Args:
        params: An object containing the `run_name` attribute, which is used as a prefix for the experiment ID.
        create_directory (bool): Whether to create a directory for the experiment. Defaults to True.

    Returns:
        tuple: A tuple containing the experiment directory path (if created) and the experiment ID.
    """
    # Get the current timestamp
    current_time = datetime.now()

    # Generate a unique experiment ID using the run name and timestamp
    experiment_id = params.run_name + current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Initialize the experiment directory variable
    experiment_directory = None

    # Create the experiment directory if requested
    if create_directory:
        experiment_directory = f'runs/{experiment_id}'
        os.makedirs(experiment_directory, exist_ok=True)

    # Return the experiment directory path and experiment ID
    return experiment_directory, experiment_id