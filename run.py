import random
from time import time
import datetime
import numpy as np
import pandas as pd
import torch
import wandb
from data_loading import Data,ImmunData
from test import test,test_xgb
from train import train_G, train_classifier,train_xgb,train_H,train_f2
from utils import get_mask,init_models,features_f_corelation,load_datasets_list,save_weights,load_weights
from visualization import visulaize_tsne, visulaize_umap
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
      self.data_type = "all"#"immunai"
      self.wandb_exp = False
      self.load_pretraind_weights = False
      self.save_weights = True
      self.iterations = 1
      self.working_models = {"F":True,"g":True,"F2":True,"F2_c":True,"H":False,"XGB":False}


def run_masks_and_vis(args):
    ## Init random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ## Conecting to device
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    if device != 'cpu':
        torch.cuda.empty_cache()
    print(f'Using device {device}')
    datasets_list = load_datasets_list(args)
    for i,f in enumerate(datasets_list):
        data = Data(data_name=f,train_ratio=args.train_ratio,features=True,all_labels=True)
        print(f"Masking dataset:{data.data_name}")
        if not os.path.exists(f"./results/{data.data_name}/"):
            os.mkdir(f"./results/{data.data_name}/")
        _,g_model_copy_f2_c = load_weights(data,device,"F2_c")
        mask_df,mask_x_df,input_df = get_mask(g_model_copy_f2_c,data,args,device)
        visulaize_umap(copy.deepcopy(mask_df),"mask_df",data.data_name)
        visulaize_umap(copy.deepcopy(mask_x_df),"mask_x_df",data.data_name)
        visulaize_umap(copy.deepcopy(input_df),"input_df",data.data_name)

        visulaize_tsne(copy.deepcopy(mask_df),"mask_df",data.data_name)
        visulaize_tsne(copy.deepcopy(mask_x_df),"mask_x_df",data.data_name)
        visulaize_tsne(copy.deepcopy(input_df),"input_df",data.data_name)


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


    ##
    datasets_list = load_datasets_list(args)
    first_data_set = True
    global_time = time()
    for i,f in enumerate(datasets_list):
        first_iteration = True
        dataset_time = time()
        ## Init WandB experiment
        if args.wandb_exp:
            wandb_exp = wandb.init(project="CellAnnotation", entity="niv_a")
            wandb_exp.name = f"Train_{f.name}"
            wandb_exp.config.update(args.__dict__)
        else: 
            wandb_exp = None
        ##

        for j in range(args.iterations):
            res_dict = {}
            res_prints = ""
            iter_time = time()
            args = arguments()

            # if args.data_type == "immunai":
            #     data = ImmunData(data_set="pbmc",genes_filter="narrow_subset",all_types=False)
            # else:
            data = Data(data_name=f,train_ratio=args.train_ratio,features=True,all_labels=True)
            print(f"Training iteration:{j} dataset:{data.data_name}")
            
            if args.working_models["F"] or args.working_models["g"]:
                cls,g_model = init_models(args=args,data=data,device=device)
                cls,cls_res_dict = train_classifier(args,device=device,data_obj=data,model=cls,wandb_exp=wandb_exp)
                args.batch_factor=4
                args.weight_decay=0
                g_model ,g_res_dict= train_G(args,device,data_obj=data,classifier=cls,model=g_model,wandb_exp=wandb_exp)

                res_dict.update(cls_res_dict)
                res_dict.update(g_res_dict)
                res_prints+="\nF Resutls\n"
                res_prints+=str(cls_res_dict)
                res_prints+="\nG Resutls\n"
                res_prints+=str(g_res_dict)
                if args.save_weights:
                    save_weights(cls=cls,g=g_model,data=data)

            args.batch_factor=1
            args.weight_decay=5e-4
            if args.working_models["F2_c"]:
                g_model_copy_f2_c = copy.deepcopy(g_model)
                f2_c,g_model_copy_f2_c,f2_c_res_dict = train_f2(args,device,data_obj=data,g_model=g_model_copy_f2_c,wandb_exp=None,model=None,concat=True)
                res_dict.update(f2_c_res_dict)
                res_prints+="\nF2_c Resutls\n"
                res_prints+=str(f2_c_res_dict)
                if args.save_weights:
                    save_weights(cls=f2_c,g=g_model_copy_f2_c,data=data,base="F2_c")
            if args.working_models["F2"]:
                g_model_copy_f2 = copy.deepcopy(g_model)
                f2,g_model_copy_f2,f2_res_dict = train_f2(args,device,data_obj=data,g_model=g_model_copy_f2,wandb_exp=None,model=None,concat=False)
                res_dict.update(f2_res_dict)
                res_prints+="\nF2 Resutls\n"
                res_prints+=str(f2_res_dict)
                if args.save_weights:
                    save_weights(cls=f2,g=g_model_copy_f2,data=data,base="F2")
            if args.working_models["H"]:
                g_model_copy_H = copy.deepcopy(g_model)
                h,g_model_copy_H,h_res_dict = train_H(args,device,data_obj=data,g_model=g_model_copy_H,wandb_exp=None,model=None)
                res_dict.update(h_res_dict)
                res_prints+="\nH Resutls\n"
                res_prints+=str(h_res_dict)
                if args.save_weights:
                    save_weights(cls=h,g=g_model_copy_H,data=data,base="H")
            if args.working_models["XGB"]:
                xgb_cls,xgb_res_dict = train_xgb(data,device)
                res_dict.update(xgb_res_dict)
                res_prints+="\nXGB Resutls\n"
                res_prints+=str(xgb_res_dict)
                if args.save_weights:
                    save_weights(cls=xgb_cls,g=None,data=data,base="XGB")
            

            # mask_df,mask_x_df,input_df = get_mask(g_model_copy_f2_c,data,args,device)
            # visulaize_umap(copy.deepcopy(mask_df),"mask_df",data.data_name)
            # visulaize_umap(copy.deepcopy(mask_x_df),"mask_x_df",data.data_name)
            # visulaize_umap(copy.deepcopy(input_df),"input_df",data.data_name)

            # visulaize_tsne(copy.deepcopy(mask_df),"mask_df",data.data_name)
            # visulaize_tsne(copy.deepcopy(mask_x_df),"mask_x_df",data.data_name)
            # visulaize_tsne(copy.deepcopy(input_df),"input_df",data.data_name)

            print(f"############### Results on {data.data_name} ############################")
            print(res_prints)
            print(f"#####################################################################")
            

            if first_iteration:
                single_data_res_df = pd.DataFrame(res_dict, index=[data.data_name])
                first_iteration = False
            else:
                single_res_df = pd.DataFrame(res_dict, index=[data.data_name])
                single_data_res_df = pd.concat([single_data_res_df, single_res_df])
            time_diff = datetime.timedelta(seconds=time()-iter_time)
            print("{}: iteration #{} took {}".format(data.data_name,j+1,time_diff))
            print(f"#################################")
        time_diff = datetime.timedelta(seconds=time()-dataset_time)
        print("{}: {} iterations took {}".format(data.data_name,args.iterations,time_diff))  
        print(f"#################################")     
        single_data_res_mean = pd.DataFrame(single_data_res_df.mean()).T
        single_data_res_mean.index = [data.data_name]
        if first_data_set:
            full_resutls_df = single_data_res_df
            mean_resutls_df = single_data_res_mean
            first_data_set = False
        else:
            full_resutls_df = pd.concat([full_resutls_df, single_data_res_df])
            mean_resutls_df = pd.concat([mean_resutls_df, single_data_res_mean])
            

    time_diff = datetime.timedelta(seconds=time()-global_time)
    print("All training took: {}".format(time_diff))   
    print(f"#################################")  
    time_for_file = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    full_resutls_df.to_csv(f"./results/{time_for_file}_full_res_df.csv")
    if args.iterations>1:
        mean_resutls_df.to_csv(f"./results/{time_for_file}_mean_res_df.csv")



if __name__ == '__main__':
    args = arguments()
    run(args)