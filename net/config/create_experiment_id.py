from datetime import datetime
import os

def create_experiment_id(params, create_directory=True):
    """
    Creates a unique experiment directory under runs/YYYY-MM-DD/, named as prefix + counter suffix.
    
    Args:
        params: Object with attribute `prefix` (experiment base name).
        create_directory (bool): If True, creates the directory.
    
    Returns:
        tuple:
            - experiment_directory (str or None): Full path like 'runs/YYYY-MM-DD/prefix_counter'
            - experiment_id (str): Unique experiment name like 'prefix' or 'prefix_1', 'prefix_2', etc.
    """

    current_time = datetime.now()
    date_str = current_time.strftime("%Y-%m-%d")
    base_dir = os.path.join("runs", date_str)
    os.makedirs(base_dir, exist_ok=True)

    base_experiment_name = params.prefix
    experiment_name = base_experiment_name
    experiment_dir = os.path.join(base_dir, experiment_name)
    counter = 1

    while os.path.exists(experiment_dir):
        experiment_name = f"{base_experiment_name}_{counter}"
        experiment_dir = os.path.join(base_dir, experiment_name)
        counter += 1

    if create_directory:
        os.makedirs(experiment_dir, exist_ok=False)

    return experiment_dir, experiment_name
