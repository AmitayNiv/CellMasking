from pkg_resources import safe_name
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from models import Classifier, G_Model


def get_mask(g_model,data_obj,args,device):
    dataset_loader = DataLoader(dataset=data_obj.all_dataset,batch_size=len(data_obj.all_dataset),shuffle=False)
    cols = list(data_obj.colnames)
    cols.append("y")
    mask_df = mask_x_df = input_df = pd.DataFrame(columns=cols)
    with torch.no_grad():
        g_model.eval()
        for X_batch, y_batch in dataset_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            mask = g_model(X_batch)
            cropped_features = X_batch * mask
            
            y = np.expand_dims(np.argmax(np.array(y_batch.detach().cpu()),axis=1), axis=1)
            mask = np.concatenate((np.array(mask.detach().cpu()),y),axis=1)
            cropped_features = np.concatenate((np.array(cropped_features.detach().cpu()),y),axis=1)
            input_x = np.concatenate((np.array(X_batch.detach().cpu()),y),axis=1)

            mask_df = pd.concat([mask_df,pd.DataFrame(mask,columns=cols)])
            mask_x_df = pd.concat([mask_x_df,pd.DataFrame(cropped_features,columns=cols)])
            input_df = pd.concat([input_df,pd.DataFrame(input_x,columns=cols)])
    
    mask_df,mask_x_df,input_df = mask_df.reset_index(),mask_x_df.reset_index(),input_df.reset_index()

    inv_mask_vals = 1-mask_df.values[:,1:-1]
    inv_mask_vals = np.concatenate(( np.expand_dims(np.array(mask_df.index),axis=1),inv_mask_vals),axis=1)
    inv_mask_vals = np.concatenate((inv_mask_vals, np.expand_dims(np.array(mask_df["y"].values),axis=1),),axis=1)
    mask_inv = pd.DataFrame(inv_mask_vals,columns=mask_df.columns)
    mask_df["label"]= mask_x_df["label"] = input_df["label"] = mask_inv["label"]= data_obj.named_labels.values
    mask_df["label_2"]= mask_x_df["label_2"] = input_df["label_2"] = mask_inv["label_2"]=data_obj.named_labels_2.values

    return mask_df,mask_x_df,input_df,mask_inv


def init_models(args,data,device):
    if args.load_cls_weights:
        print("Loading pre-trained weights for classifier")
        cls = torch.load(r"/media/data1/nivamitay/CellMasking/weights/cls.pt").to(device)
    else:
        print("Initializing classifier")
        cls = Classifier(data.n_features ,dropout=args.dropout,number_of_classes=data.number_of_classes,first_division=2)
        cls = cls.to(device)
    
    if args.load_g_weights:
        print("Loading pre-trained weights for G model")
        g_model = torch.load(r"/media/data1/nivamitay/CellMasking/weights/g_model.pt").to(device)
    else:
        print("Initializing G model")
        g_model = G_Model(data.n_features,first_division=2)
        g_model = g_model.to(device)

    return cls,g_model
        