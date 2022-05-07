from pkg_resources import safe_name
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from models import Classifier, G_Model
import matplotlib.pyplot as plt
import copy
import os

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


    mask_df["label"]= mask_x_df["label"] = input_df["label"] = data_obj.named_labels.values
    # mask_df["label_2"]= mask_x_df["label_2"] = input_df["label_2"] =data_obj.named_labels_2.values

    return mask_df,mask_x_df,input_df


def init_models(args,data,device,base = ""):
    base = base+"_" if base != "" else base
    if args.load_pretraind_weights:
        cls,g_model = load_weights(cls,g_model)
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
        
        plt.savefig(r"/media/data1/nivamitay/CellMasking/results/f_diff.png")


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
    base = base+"_" if base != "" else base
    if not os.path.exists(f"./weights/{data.data_name}/"):
        os.mkdir(f"./weights/{data.data_name}/")
    torch.save(cls,f"./weights/{data.data_name}/{base}cls.pt")
    torch.save(g,f"./weights/{data.data_name}/{base}g.pt")
    print(f"{base} Models was saved to ./weights/{data.data_name}")


def load_weights(data,device,base = ""):
    print("Loading pre-trained weights for classifier")
    cls = torch.load(f"./weights/{data.data_name}/{base}cls.pt").to(device)
    print("Loading pre-trained weights for G model")
    g_model = torch.load(f"./weights/{data.data_name}/{base}g.pt").to(device)
    return cls,g_model

