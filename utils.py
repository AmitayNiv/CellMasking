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
import eli5


def get_mask(g_model,data_obj,args,device,bin_mask=False):
    dataset_loader = DataLoader(dataset=data_obj.all_dataset,batch_size=len(data_obj.all_dataset)//8,shuffle=False)
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

    return mask_arr

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
    if base =="XGB":
        print(f"Loading pre-trained weights for {base} classifier")
        cls = xgb.XGBClassifier(objective="multi:softproba")
        cls.load_model(f"./weights/1500_genes_weights/{data.data_name}/{base}.json")
        g_model = None
    elif base =="RF":
        print(f"Loading pre-trained weights for {base} classifier")
        cls = joblib.load(f"./weights/1500_genes_weights/{data.data_name}/{base}.joblib")
        g_model = None
    else:
        if only_g:
            cls = None
        else:
            print(f"Loading pre-trained weights for {base} classifier")
            cls = torch.load(f"./weights/1500_genes_weights/{data.data_name}/{base_print}cls.pt")#.to(device)
        print(f"Loading pre-trained weights for {base} G model")
        g_model = torch.load(f"./weights/1500_genes_weights/{data.data_name}/{base_print}g.pt").to(device)
    return cls,g_model


def concat_average_dfs(aux2,aux3):
    # Putting the same index together
#     I use the try because I want to use this function recursive and 
#     I could potentially introduce dataframe with those indexes. This
#     is not the best way.
    try:
        aux2.set_index(['feature', 'target'],inplace = True)
    except:
        pass
    try:
        aux3.set_index(['feature', 'target'],inplace = True)
    except:
        pass
    # Concatenating and creating the meand
    aux = pd.DataFrame(pd.concat([aux2['weight'],aux3['weight']]).groupby(level = [0,1]).mean())
    # Return in order
    #return aux.sort_values(['weight'],ascending = [False],inplace = True)
    return aux


def get_tree_explaination(data):
    cols = data.colnames
    cols.append("y")
    rf_important = pd.DataFrame(columns=cols)
    ###############################################
    rf_model = joblib.load(f"./weights/1500_genes_weights/{data.data_name}/RF.joblib")
    X = np.array(data.all_dataset.X_data)

    y = np.array(data.all_dataset.y_data)
    y = np.argmax(y,axis=1)
    for sample in range(X.shape[0]):
        aux1 = eli5.sklearn.explain_prediction.explain_prediction_tree_classifier(rf_model,X[sample],feature_names=np.array(data.colnames),targets=[y[sample]])
        aux1 = eli5.format_as_dataframe(aux1).drop(0)
        sample_important = pd.DataFrame(np.zeros((1,len(cols))),columns=cols)
        sample_important[aux1.feature.values]=aux1.weight.values
        sample_important["y"] = y[sample]

        rf_important = pd.concat([rf_important,sample_important],axis=1)
    return rf_important

             
