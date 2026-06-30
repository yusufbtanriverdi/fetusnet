import wandb

def initialize_wandb(params, exp_id):
    """ Initializes wandb for selected parameters. """
    wandb.init(
            # set the wandb project where this run will be logged
            project=params.wandb.wandbpro,
            # track hyperparameters and run metadata
            config=params,
            # name for run
            name=exp_id
)
    
