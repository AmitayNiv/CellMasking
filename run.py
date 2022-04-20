import random
import numpy as np
import torch
import wandb
from data_loading import Data,ImmunData
from test import test,test_xgb
from train import train_G, train_classifier,train_xgb,train_H
from utils import get_mask


CUDA_VISIBLE_DEVICES=4

class arguments:
   def __init__(self):
      self.seed = 3407
      self.cls_epochs = 70
      self.g_epochs = 20
      self.cls_lr = 0.002
      self.g_lr = 0.0002
      self.weight_decay=5e-4
      self.dropout=0
      self.batch_size = 50
      self.batch_factor = 1
      self.train_ratio = 0.7
      self.data_type = "other"#"immunai"
      self.save_cls_checkpoints = True
      self.save_g_checkpoints = False




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

    ## Init WandB experiment
    # wandb_exp = wandb.init(project="CellAnnotation", entity="niv_a")
    # wandb_exp.name = f"Train"
    # wandb_exp.config.update(
    # dict(cls_epochs=args.cls_epochs,g_epochs= args.g_epochs,cls_lr=args.cls_lr,g_lr=args.g_lr, \
    # batch_size=args.batch_size, train_ratio=args.train_ratio, weight_decay=args.weight_decay))

    ##
    if args.data_type == "immunai":
        data = ImmunData(data_set="pbmc",genes_filter="narrow_subset",all_types=False)
    else:
        data = Data(train_ratio=args.train_ratio,features=True)
    # data_test = Data(train_ratio=args.train_ratio,features=True,data_name='10X_pbmc_5k_nextgem.h5ad',test_set=True)
    
    
    cls = train_classifier(args,device=device,data_obj=data,model=None,wandb_exp=None)
    if args.save_cls_checkpoints:
        torch.save(cls,r"/media/data1/nivamitay/CellMasking/weights/cls.pt")
    g_model = train_G(args,device,data_obj=data,classifier=cls,model=None,wandb_exp=None)
    # g_model = train_G(args,device,data_obj=data,classifier=cls_z,model=None,wandb_exp=None)
    if args.save_g_checkpoints:
        torch.save(g_model,r"/media/data1/nivamitay/CellMasking/weights/g_model.pt")

    h_cls =  train_H(args,device,data_obj=data,g_model=g_model,wandb_exp=None,model=None)
    
    # test(cls,g_model=g_model,device=device,data_obj=data_test)
    # xgb_cls = train_xgb(data,device)
    # test_xgb(xgb_cls,data_test,device)

    # mask_df,mask_x_df,input_df = get_mask(g_model,data,args,device)
    # mask_df["label"]= mask_x_df["label"] = input_df["label"] = data.named_labels.values
    # # mask_df = mask_df.groupby(by=["label"]).sum()
    # mask_df.to_csv( r"/media/data1/nivamitay/CellMasking/results/mask.csv")
    # mask_x_df.to_csv( r"/media/data1/nivamitay/CellMasking/results/mask_x_df.csv")
    # input_df.to_csv( r"/media/data1/nivamitay/CellMasking/results/input_df.csv")
    print()







    



if __name__ == '__main__':
    args = arguments()
    run(args)