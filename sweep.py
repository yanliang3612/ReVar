import wandb
from src.argument import parse_args_sweep
from src.utils import set_random_seeds
from models import RVGNN_Trainer
import os

import torch

START_SWEEP = True
RUN_SWEEP = False


def sweep_fun():
    with wandb.init(project='Imbalance', settings=wandb.Settings(start_method='fork')) as run:
        # torch.use_deterministic_algorithms(True)
        # set default configuration
        run.config.setdefaults(default_config)
        # start training
        set_random_seeds(0)
        torch.set_num_threads(2)
    
        args = run.config
        embedder = RVGNN_Trainer(args)
        train_summary = embedder.train()
        wandb.log(train_summary)


if __name__ == '__main__':
    # define the base dict
    default_config = parse_args_sweep()
    wandb.agent(sweep_id=default_config.sweep_id, function=sweep_fun)
    sweep_fun()

