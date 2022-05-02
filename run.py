import random
import numpy as np
import pandas as pd
import torch
import wandb
from data_loading import Data,ImmunData
from test import test,test_xgb
from train import train_G, train_classifier,train_xgb,train_H,train_f2
from utils import get_mask,init_models,features_f_corelation
from visualization import visulaize_tsne
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
      self.data_type = "other"#"immunai"
      self.wandb_exp = False
      self.load_cls_weights = False
      self.load_g_weights = False
      self.save_cls_checkpoints = False
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
    if args.wandb_exp:
        wandb_exp = wandb.init(project="CellAnnotation", entity="niv_a")
        wandb_exp.name = f"Train"
        wandb_exp.config.update(args.__dict__)
    else: 
        wandb_exp = None
    # dict(cls_epochs=args.cls_epochs,g_epochs= args.g_epochs,cls_lr=args.cls_lr,g_lr=args.g_lr, \
    # batch_size=args.batch_size, train_ratio=args.train_ratio, weight_decay=args.weight_decay))

    ##

    # data_test = Data(train_ratio=args.train_ratio,features=True,data_name='10X_pbmc_5k_nextgem.h5ad',test_set=True)
    first = True
    for i,f in enumerate(os.scandir(r"/media/data1/nivamitay/CellMasking/data/singleCell/")):
        args = arguments()
        if f.name == 'features.csv':
            continue
        if args.data_type == "immunai":
            data = ImmunData(data_set="pbmc",genes_filter="narrow_subset",all_types=False)
        else:
            data = Data(data_name=f.name,train_ratio=args.train_ratio,features=True,all_labels=True)

        cls,g_model = init_models(args=args,data=data,device=device)
        cls,cls_res_dict = train_classifier(args,device=device,data_obj=data,model=cls,wandb_exp=wandb_exp)
        args.batch_factor=4
        args.weight_decay=0
        g_model ,g_res_dict= train_G(args,device,data_obj=data,classifier=cls,model=g_model,wandb_exp=wandb_exp)


        if args.save_cls_checkpoints:
            torch.save(cls,r"/media/data1/nivamitay/CellMasking/weights/cls.pt")
        if args.save_g_checkpoints:
            torch.save(g_model,r"/media/data1/nivamitay/CellMasking/weights/g_model.pt")


        args.cls_epochs=10
        args.g_epochs = 10
        args.batch_factor=1
        args.weight_decay=5e-4
        args.dropout=0.2
        args.cls_lr = 0.002
        g_model_copy_1 = copy.deepcopy(g_model)
        g_model_copy_2 = copy.deepcopy(g_model)
        g_model_copy_3 = copy.deepcopy(g_model)
        f2,f2_c_res_dict = train_f2(args,device,data_obj=data,g_model=g_model_copy_1,wandb_exp=None,model=None,concat=True)
        f2,f2_res_dict = train_f2(args,device,data_obj=data,g_model=g_model_copy_2,wandb_exp=None,model=None,concat=False)
        h_cls,h_res_dict=  train_H(args,device,data_obj=data,g_model=g_model_copy_3,wandb_exp=None,model=None)
        # xgb_cls,xgb_res_dict = train_xgb(data,device)
        # features_f_corelation(args,device,data_obj=data,g_model=g_model,cls=cls)
        print()
        print(f"############### Results on {data.data_name} ############################")
        print("cls Results")
        print(cls_res_dict)
        print("G Results")
        print(g_res_dict)
        print("F2_C Results")
        print(f2_c_res_dict)
        print("F2 Results")
        print(f2_res_dict)
        print("H Results")
        print(h_res_dict)
        # print("XGB Results")
        # print(xgb_res_dict)
        print(f"#####################################################################")
        
        # test(cls,g_model=g_model,device=device,data_obj=data_test)
        
        cls_res_dict.update(g_res_dict)
        cls_res_dict.update(f2_c_res_dict)
        cls_res_dict.update(f2_res_dict)
        cls_res_dict.update(h_res_dict)
        # cls_res_dict.update(xgb_res_dict)
        if first:
            res_df = pd.DataFrame(cls_res_dict, index=[data.data_name])
            first = False
        else:
            single_res_df = pd.DataFrame(cls_res_dict, index=[data.data_name])
            res_df = pd.concat([res_df, single_res_df])
        


    
    res_df.to_csv(r"/media/data1/nivamitay/CellMasking/results/res_df_iter2.csv")

        # test_xgb(xgb_cls,data_test,device)

    #mask_df,mask_x_df,input_df,mask_inv = get_mask(g_model,data,args,device)


    # visulaize_tsne(mask_df,"mask_df",wandb_exp)
    # visulaize_tsne(mask_x_df,"mask_x_df",wandb_exp)
    # visulaize_tsne(input_df,"input_df",wandb_exp)
    # visulaize_tsne(mask_inv,"mask_inv",wandb_exp)


    # mask_inv.to_csv( r"/media/data1/nivamitay/CellMasking/results/mask_inv_wide.csv")
    # mask_df.to_csv( r"/media/data1/nivamitay/CellMasking/results/mask_wide.csv")
    # mask_x_df.to_csv( r"/media/data1/nivamitay/CellMasking/results/mask_x_wide.csv")
    # input_df.to_csv( r"/media/data1/nivamitay/CellMasking/results/input_wide.csv")

    # print()






if __name__ == '__main__':
    args = arguments()
    run(args)