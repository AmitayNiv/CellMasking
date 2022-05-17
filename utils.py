from pkg_resources import safe_name
from pyparsing import col
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from models import Classifier, G_Model
import matplotlib.pyplot as plt
import copy
import os
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import gseapy as gp
from data_loading import Data
from time import time
import datetime

def get_mask(g_model,data_obj,args,device,bin_mask=False):
    dataset_loader = DataLoader(dataset=data_obj.all_dataset,batch_size=len(data_obj.all_dataset)//8,shuffle=False)
    cols = list(data_obj.colnames)
    cols.append("y")
    print(f"Creating mask for {data_obj.data_name}")
    first_batch = True
    with torch.no_grad():
        g_model.eval()
        for X_batch, y_batch in dataset_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            mask = g_model(X_batch)
            if bin_mask:
                mask = torch.where(mask>0.5,1,0)
            mask.requires_grad = False
            if first_batch:
                mask_arr = mask
                first_batch = False
            else:
                mask_arr = torch.cat((mask_arr,mask), 0)
                

    # mask_df["label"] = data_obj.named_labels.values
    # if hasattr(data_obj,"patient"):
    #     mask_df["patient"] = data_obj.patient.values
    mean = mask_arr.mean(dim=0)
    ten_p = torch.quantile(mean,0.85)
    val = torch.where(mean>ten_p,mean,torch.std(mask_arr, dim=0))
    return  val#torch.var(mask_df, dim=0)#torch.max(mask_df.mean(dim=0),3*torch.std(mask_df, dim=0))#torch.quantile(mask_df.detach().cpu(),0.9,dim=0)

def get_mask_and_mult(g_model,data_obj,args,device,bin_mask=False):
    dataset_loader = DataLoader(dataset=data_obj.all_dataset,batch_size=len(data_obj.all_dataset)//8,shuffle=False)
    cols = list(data_obj.colnames)
    double_cols = copy.deepcopy(cols)
    double_cols.extend(cols)
    double_cols.append("y")
    cols.append("y")
    mask_df = input_df = pd.DataFrame(columns=cols)
    mask_x_df = pd.DataFrame(columns=double_cols)
    print(f"creating mask for {data_obj.data_name}")
    with torch.no_grad():
        g_model.eval()
        for X_batch, y_batch in dataset_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            mask = g_model(X_batch)
            if bin_mask:
                mask = torch.where(mask>0.5,1,0)
            cropped_features = X_batch*mask
            X_test_batch_bin = torch.where(X_batch==0, 1, 0)

            cropped_features_neg = X_test_batch_bin *mask
            cropped_features = torch.concat((cropped_features,cropped_features_neg),dim=1)

            
            y = np.expand_dims(np.argmax(np.array(y_batch.detach().cpu()),axis=1), axis=1)
            mask = np.concatenate((np.array(mask.detach().cpu()),y),axis=1)
            cropped_features = np.concatenate((np.array(cropped_features.detach().cpu()),y),axis=1)
            input_x = np.concatenate((np.array(X_batch.detach().cpu()),y),axis=1)

            mask_df = pd.concat([mask_df,pd.DataFrame(mask,columns=cols)])
            mask_x_df = pd.concat([mask_x_df,pd.DataFrame(cropped_features,columns=double_cols)])
            input_df = pd.concat([input_df,pd.DataFrame(input_x,columns=cols)])
    
    mask_df,mask_x_df,input_df = mask_df.reset_index(),mask_x_df.reset_index(),input_df.reset_index()



    mask_df["label"]= mask_x_df["label"] = input_df["label"] = data_obj.named_labels.values
    if hasattr(data_obj,"patient"):
        mask_df["patient"]= mask_x_df["patient"] = input_df["patient"] = data_obj.patient.values

    return mask_df,mask_x_df,input_df


def init_models(args,data,device,base = ""):
    if args.load_pretraind_weights:
        cls,g_model = load_weights(data,device,base)
    else:
        print("Initializing classifier")
        cls = Classifier(data.n_features ,dropout=args.dropout,number_of_classes=data.number_of_classes,first_division=2)
        cls = cls.to(device)
        print("Initializing G model")
        g_model = G_Model(data.n_features,first_division=2)
        g_model = g_model.to(device)
    return cls,g_model
    

def features_f_corelation(args,device,data_obj,g_model,cls):
    dataset_loader = DataLoader(dataset=data_obj.all_dataset,batch_size=1,shuffle=False)
    with torch.no_grad():
        g_model.eval()
        cls.eval()
        j=0
        g_scroe = []
        pred_diff = []
        for X_batch, y_batch in dataset_loader:
            j+=1
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            mask = g_model(X_batch)

            for i in range(X_batch.shape[1]):
                if X_batch[0,i]!=0:
                    y_reg_pred = torch.softmax(cls(X_batch), dim = 1)
                    first_label = y_reg_pred.argmax().item()
                    y_reg_pred = y_reg_pred[0,first_label].item()
                    X_batch_copy = copy.deepcopy(X_batch)
                    X_batch_copy[0,i] = 0
                    y_zer_pred = torch.softmax(cls(X_batch_copy), dim = 1)
                    y_zer_pred = y_zer_pred[0,first_label].item()
                    pred_diff.append(y_reg_pred-y_zer_pred)
                    g_scroe.append(mask[0,i].item())
            print(j)
        plt.plot(pred_diff,g_scroe,"o")
        plt.xlabel("prediction diff")
        plt.ylabel("G score")
        
        plt.savefig(r"./results/f_diff.png")


    pass


def load_datasets_list(args):
    datasets_list = []
    for i,f in enumerate(os.scandir(r"./data/singleCell/")):
        if f.name == 'features.csv':
            continue
        if args.data_type == "all" or f.name == args.data_type +".h5ad":
            datasets_list.append(f)
                
    return datasets_list


def save_weights(cls,g,data,base = ""):
    base_print = base+"_" if base != "" else base
    if not os.path.exists(f"./weights/1500_genes_weights/{data.data_name}/"):
        os.mkdir(f"./weights/1500_genes_weights/{data.data_name}/")
    if base=="XGB":
        cls.save_model(f"./weights/1500_genes_weights/{data.data_name}/{base}.json")
    elif base=="RF":
        joblib.dump(cls, f"./weights/1500_genes_weights/{data.data_name}/{base}.joblib")
    else:
        torch.save(cls,f"./weights/1500_genes_weights/{data.data_name}/{base_print}cls.pt")
        torch.save(g,f"./weights/1500_genes_weights/{data.data_name}/{base_print}g.pt")
    print(f"{base} Models was saved to ./weights/1500_genes_weights/{data.data_name}")


def load_weights(data,device,base = "",only_g=False):
    base_print = base+"_" if base != "" else base
    print(f"Loading pre-trained weights for {base} classifier")
    if base =="XGB":
        cls = xgb.XGBClassifier(objective="multi:softproba")
        cls.load_model(f"./weights/{data.data_name}/{base}.json")
        g_model = None
    elif base =="RF":
        cls = joblib.load(f"./weights/{data.data_name}/{base}.joblib")
        g_model = None
    else:
        if only_g:
            cls = None
        else:
            cls = torch.load(f"./weights/1500_genes_weights/{data.data_name}/{base_print}cls.pt")#.to(device)
        print(f"Loading pre-trained weights for {base} G model")
        g_model = torch.load(f"./weights/1500_genes_weights/{data.data_name}/{base_print}g.pt").to(device)
    return cls,g_model


def run_gsea(args,device):
    datasets_list = load_datasets_list(args)
    # with open("./data/immunai_data_set.gmt")as gmt:
    cols = ["Data","Model","nes","pval","fdr"]
    results_df = pd.DataFrame(columns=cols)
    global_time = time()
    for i,f in enumerate(datasets_list):
        dataset_time = time()
        print(f"\n### Starting work on {f.name[:-5]} ###")
        data = Data(data_inst=f,train_ratio=args.train_ratio,features=True,all_labels=False,test_set=True)
        for mod in ["G","F2_c","F2"]:
            base_print = "" if mod =="G" else mod
            _,g = load_weights(data,device,base_print,only_g=True)
            mask_df = get_mask(g,data,args,device)

            rnk = pd.DataFrame(columns=["0","1"])
            rnk["0"] = data.colnames
            rnk["1"] = mask_df.cpu()
            rnk = rnk.sort_values(by="1",ascending=False)
            pre_res = gp.prerank(rnk=rnk, gene_sets=f'./data/gmt_files/all.gmt',
                    processes=4,
                    permutation_num=100, # reduce number to speed up testing
                    no_plot =True,
                    outdir=f'./results/prerank/{f.name[:-5]}/prerank_report_all', format='png', seed=6,min_size=1, max_size=600)
            res_list = [data.data_name,mod,pre_res.res2d["nes"].values[0],pre_res.res2d["pval"].values[0],pre_res.res2d["fdr"].values[0]]
            single_res_df =pd.DataFrame([res_list],columns=cols)
            results_df = pd.concat([results_df, single_res_df])

        xgb_cls = xgb.XGBClassifier(objective="multi:softproba")
        xgb_cls.load_model(f"./weights/1500_genes_weights/{data.data_name}/XGB.json")
        xgb_rank = pd.DataFrame(columns=["0","1"])
        xgb_rank["0"] = data.colnames
        xgb_rank["1"] = xgb_cls.feature_importances_
        xgb_rank = xgb_rank.sort_values(by="1",ascending=False)
        pre_res_xgb = gp.prerank(rnk=xgb_rank, gene_sets=f'./data/gmt_files/all.gmt',
            processes=4,
            permutation_num=100, # reduce number to speed up testing
            no_plot =True,
            outdir=f'./results/prerank/{f.name[:-5]}/prerank_report_all_xgb', format='png', seed=6,min_size=1, max_size=600)
        res_list = [data.data_name,"XGB",pre_res_xgb.res2d["nes"].values[0],pre_res_xgb.res2d["pval"].values[0],pre_res_xgb.res2d["fdr"].values[0]]
        single_res_df =pd.DataFrame([res_list],columns=cols)
        results_df = pd.concat([results_df, single_res_df])

        ###############################################
        rf_model = joblib.load(f"./weights/1500_genes_weights/{data.data_name}/RF.joblib")
        rf_rank = pd.DataFrame(columns=["0","1"])
        rf_rank["0"] = data.colnames
        rf_rank["1"] = rf_model.feature_importances_
        rf_rank = rf_rank.sort_values(by="1",ascending=False)
        pre_res_rf = gp.prerank(rnk=rf_rank, gene_sets=f'./data/gmt_files/all.gmt',
            processes=4,
            permutation_num=100, # reduce number to speed up testing
            no_plot =True,
            outdir=f'./results/prerank/{f.name[:-5]}/prerank_report_all_rf', format='png', seed=6,min_size=1, max_size=600)
        res_list = [data.data_name,"RF",pre_res_rf.res2d["nes"].values[0],pre_res_rf.res2d["pval"].values[0],pre_res_rf.res2d["fdr"].values[0]]
        single_res_df =pd.DataFrame([res_list],columns=cols)
        results_df = pd.concat([results_df, single_res_df])
        time_diff = datetime.timedelta(seconds=time()-dataset_time)
        print("Working on {}:took {}".format(data.data_name,time_diff))
        print(f"#################################")  

    results_df.to_csv("./results/prerank/prerank_res_df_std.csv")
    time_diff = datetime.timedelta(seconds=time()-global_time)
    print("All training took: {}".format(time_diff))   
    print(f"#################################")  

    
    print()

def run_heatmap_procces(args,device):
    datasets_list = load_datasets_list(args)
    for i,f in enumerate(datasets_list):
        dataset_time = time()
        print(f"\n### Starting work on {f.name[:-5]} ###")
        data = Data(data_inst=f,train_ratio=args.train_ratio,features=True,all_labels=False,test_set=True)
        _,g_model = load_weights(data,device,"F2_c",only_g=True)
        dataset_loader = DataLoader(dataset=data.all_dataset,batch_size=len(data.all_dataset)//8,shuffle=False)
        cols = list(data.colnames)
        cols.append("y")
        mask_df = pd.DataFrame(columns=cols)
        print(f"Creating mask for {data.data_name}")
        first_batch = True
        with torch.no_grad():
            g_model.eval()
            for X_batch, y_batch in dataset_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                mask = g_model(X_batch)
                mask.requires_grad = False

                y = np.expand_dims(np.argmax(np.array(y_batch.detach().cpu()),axis=1), axis=1)
                mask = np.concatenate((np.array(mask.detach().cpu()),y),axis=1)
            
                mask_df = pd.concat([mask_df,pd.DataFrame(mask,columns=cols)])
            mask_df= mask_df.reset_index()



            mask_df["label"] = data.named_labels.values
            if hasattr(data,"patient"):
                mask_df["patient"] = data.patient.values




        ten_p = np.quantile(mask_df[data.colnames].values.mean(axis=0),0.9)

        max_patient = data.full_data.obs.patient.value_counts()[:40].index
        mask_df = mask_df[mask_df['patient'].isin(max_patient)]
        mask_df['patient'] = np.array(mask_df['patient'])

        data_types = ["naive CD8","memory CD8","naive CD4","memory CD4"]
        for current_data_name in data_types:
            current_data = mask_df[mask_df["label"]==current_data_name]
            current_data_mean = current_data.groupby("patient")[data.colnames].agg(np.mean)
            current_data_mean.columns = data.colnames
            current_data_std = current_data.groupby("patient")[data.colnames].agg(np.std)
            current_data_std.columns = data.colnames

            # arr = current_data_std.values
            # current_data_mean_mean = current_data_mean.mean(axis=0)
            # best_genes = current_data_mean_mean[current_data_mean_mean>current_data_mean_mean.quantile(0.9)].index
            # arr[current_data_mean.values>ten_p] = current_data_mean.values[current_data_mean.values>ten_p]
            # df = pd.DataFrame(arr,columns=data.colnames)
            current_data_mean_mean = current_data_mean.mean(axis=0)
            best_genes = current_data_mean_mean[current_data_mean_mean>ten_p].index
            df = current_data_mean[best_genes]

            plt.figure(figsize=(30,20))
            plt.imshow(df.values,cmap="hot")
            plt.xticks(np.arange(0.5, len(best_genes), 1), best_genes,rotation = 90)
            plt.yticks(np.arange(0.5, df.shape[0], 1), current_data_mean.index)
            plt.title(current_data_name)
            plt.colorbar()
            plt.savefig(f"./results/heatmap_{current_data_name}.png")




        print()


                    
