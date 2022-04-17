import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


def get_mask(g_model,data_obj,args,device):
    dataset_loader = DataLoader(dataset=data_obj.all_dataset,batch_size=args.batch_size,shuffle=False)
    mask_df = pd.DataFrame(columns=data_obj.colnames)
    with torch.no_grad():
        g_model.eval()
        for X_batch, y_batch in dataset_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            mask = (np.array(g_model(X_batch).detach().cpu())>0.5).astype(int)
            mask_df = pd.concat([mask_df,pd.DataFrame(mask,columns=data_obj.colnames)])

    return mask_df


