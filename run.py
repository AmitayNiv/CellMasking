import random
import numpy as np
import torch
import wandb
from data_loading import Data


class arguments:
   def __init__(self):
      self.seed = 3407
      self.cls_epochs = 50
      self.g_epochs=20
      self.cls_lr=0.00002
      self.g_lr = 0.5
      self.weight_decay=5e-4
      self.dropout=0
      self.optim = "SGD"
      self.batch_size = 50
      self.train_ratio = 0.6



def run(args):
    ## Init random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ## Conecting to device
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    if device != 'cpu':
        torch.cuda.empty_cache()
    print(f'Using device {device}')

    ## Init WandB experiment
    wandb_exp = wandb.init(project="CellAnnotation", entity="niv_a")
    wandb_exp.name = f"Train"
    wandb_exp.config.update(
    dict(cls_epochs=args.cls_epochs,g_epochs= args.g_epochs,cls_lr=args.cls_lr,g_lr=args.g_lr, \
    batch_size=args.batch_size, train_ratio=args.train_ratio, weight_decay=args.weight_decay))






    



if __name__ == '__main__':
    args = arguments()
    data = Data()
    run(args)