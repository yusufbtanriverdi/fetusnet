import wandb

def initialize_wandb(params, fold=0):
    """ Initializes wandb for selected parameters. """
    wandb.init(
            # set the wandb project where this run will be logged
            project=params.wandbpro,
            # track hyperparameters and run metadata
            config=params,
            # name for run
            name=f'{params.prefix}_{params.lmks[0]}_fold{fold}',)
    
