import wandb

def initialize_wandb(params):
    """ Initializes wandb for selected parameters. """
    wandb.init(
            # set the wandb project where this run will be logged
            project=params.wandbpro,
            # track hyperparameters and run metadata
            config=params,
            # name for run
            # TODO: take this as argument and test.
            name=params.run_name
        )
    
