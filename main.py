from src.argument import parse_args
from src.utils import set_random_seeds
from models import RVGNN_Trainer
import os
import pathlib
import torch
import json


USE_WANDB = False

def main():
    args = parse_args()
    set_random_seeds(0)
    torch.set_num_threads(2)
    embedder = RVGNN_Trainer(args)
    train_summary = embedder.train()
    if USE_WANDB:

        import wandb

        # it will log args automatically
        wandb.init(project=args.project, settings=wandb.Settings(start_method='fork'),
                   config=args)
        #
        wandb.log(train_summary)
    else:
        from src.utils import config2string
        config_log = config2string(args)
        summary_log = train_summary['log_txt']
        pathlib.Path('logs').mkdir(exist_ok=True)
        with open(os.path.join('.', 'logs', args.project), 'a') as f:
            print("\n[Config] {}\n".format(config_log), file=f)
            # print(summary_log, file=f)
            print(summary_log)
if __name__ == "__main__":
    main()
