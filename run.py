import random
from time import time
import datetime
import numpy as np
import pandas as pd
import torch
import wandb
from run_tasks import run_train,run_masks_creation,run_masks_and_vis,run_gsea,run_heatmap_procces,run_per_sample_gsea
import os
import copy

CUDA_VISIBLE_DEVICES=4

class arguments(object):
   def __init__(self):
      self.seed = 3407
      self.cls_epochs = 10
      self.g_epochs = 10
      self.cls_lr = 0.002
      self.g_lr = 0.0002
      self.weight_decay=5e-4
      self.dropout=0.2
      self.batch_size = 50
      self.batch_factor = 1
      self.train_ratio = 0.7
      self.data_type = "all"#"immunai"# "10X_pbmc_5k_v3"
      self.wandb_exp = False
      self.load_pretraind_weights = False
      self.save_weights = False
      self.iterations = 5
      self.working_models = {"F":True,"g":True,"F2":False,"F2_c":True,"H":False,"XGB":True,"RF":True}
      self.task = "Train"




def run(args):
    ## Init random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ## Conecting to device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device != 'cpu':
        torch.cuda.empty_cache()
    print(f'Using device {device}')
    # run_gsea(args)
    if args.task =="Train":
        # run_masks_creation(args=args,device=device)
        # run_per_sample_gsea(args,device)
        run_train(args,device)
        # 








if __name__ == '__main__':
    args = arguments()
    run(args)