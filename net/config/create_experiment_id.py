from datetime import datetime
import os 

def create_experiment_id(params, create_directory = True):


    current_time = datetime.now()
    experiment_id = params.run_name + current_time.strftime("%Y-%m-%d_%H-%M-%S")

    if create_directory:
        experiment_directory = f'runs/{experiment_id}'
        os.makedirs(experiment_directory, exist_ok=True)

    return experiment_directory, experiment_id