import random
from time import time
import datetime
import numpy as np
import pandas as pd
import torch
import wandb
from run_tasks import run_train,run_create_and_save_masks,run_masks_and_vis,\
run_gsea,run_heatmap_procces,run_per_sample_gsea_compare,run_per_sample_gsea,run_global_feature_selection,run_test
from train import train_mnist
import os
import copy

CUDA_VISIBLE_DEVICES=4

class arguments(object):
   def __init__(self):
      self.seed = 3407
      self.cls_epochs = 20
      self.g_epochs = 20
      self.cls_lr = 0.002
      self.g_lr = 0.0002
      self.weight_decay = 5e-4
      self.dropout = 0.2
      self.batch_size = 2048
      self.batch_factor = 1
      self.train_ratio = 0.7
      self.data_type =  "rat_aging_cell_atlas_ma_2020"
      self.wandb_exp = False
      self.load_pretraind_weights = False
      self.save_weights = False
      self.iterations = 1
      self.working_models = {"F":True,"g":True,"F2":True,"H":True,"XGB":False,"RF":False}
      self.task = "train_mnist"#"Feature Selection"#"Train"




def run(args):
    ## Init random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ## Conecting to device
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    if device != 'cpu':
        torch.cuda.empty_cache()
    print(f'Using device {device}')
    if args.task =="Train":
        print("Starting Train")
        run_train(args,device)
    elif args.task =="Mask Creation":
        print("Starting Mask Creation")
        run_create_and_save_masks(args,device)
    elif args.task =="Masks Visualizatin":
        print("Starting Masks Visualizatin")
        run_masks_and_vis(args,device)
    elif args.task =="GSEA":
        print("Starting GSEA Analisys")
        run_gsea(args,device)
    elif args.task =="Heatmaps":
        print("Starting Important Heatmaps Calculation")
        run_heatmap_procces(args,device)
    elif args.task =="GSEA per Sample Compariosn":
        print("Starting GSEA per Sample Compariosn")
        run_per_sample_gsea_compare(args,device)
    elif args.task =="GSEA per Sample":
        print("Starting GSEA per Sample")
        run_per_sample_gsea(args,device)
    elif args.task == "Feature Selection":
        print("Starting Features Filtering")
        run_global_feature_selection(args,device)
    elif args.task =="Test":
        run_test(args,device)
    elif args.task =="train_mnist":
        train_mnist(args,device)

        


if __name__ == '__main__':
    args = arguments()
    run(args)


    index = 0
    fig =plt.figure(figsize=(10,10))
    plt.imshow(X_test_batch[index].view(28,28).cpu(), cmap='hot', interpolation='none')
    rgb = np.stack([(mask_test[index]>0.95).view(28,28).T.detach().cpu(),np.zeros((28,28)),np.zeros((28,28))], axis=2)
    print(torch.argwhere(mask_test[index]>0.95).view(-1))
    rgb = rgb.astype(float)
    plt.imshow(rgb, cmap='hot', alpha=0.8)
    fig.savefig("fig3.png")