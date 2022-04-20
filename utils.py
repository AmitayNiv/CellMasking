import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


def get_mask(g_model,data_obj,args,device):
    dataset_loader = DataLoader(dataset=data_obj.all_dataset,batch_size=args.batch_size,shuffle=False)
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

    return mask_df.reset_index(),mask_x_df.reset_index(),input_df.reset_index()


