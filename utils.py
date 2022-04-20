import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.manifold import TSNE

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


def visulaize_tsne(data_set_name):
    data_set_name_csv =data_set_name+".csv"
    res_folder_path = r'c:\Users\niv.a\Documents\GitHub\CellMasking\CellMasking\results'
    df_path = os.path.join(res_folder_path,data_set_name_csv)
    data_set = pd.read_csv(df_path,index_col=0)

    data_set = data_set[data_set["y"]==3.0]
    feat_cols = data_set.columns[1:-2]
    data_subset = data_set[feat_cols].values

    plt.figure(figsize=(16,10))
    plt.imshow(data_subset,cmap="gray")
    data_set_name_png =data_set_name+"_img.png"
    plt.savefig(os.path.join(res_folder_path,data_set_name_png))


    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
    tsne_results = tsne.fit_transform(data_subset)
    data_set['tsne-2d-one'] = tsne_results[:,0]
    data_set['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="label",
    palette=sns.color_palette("hls",len(np.unique(data_set.y.values))),
    data=data_set,
    legend="full",
    alpha=0.3).set(title=data_set_name)
    data_set_name_png =data_set_name+".png"
    plt.savefig(os.path.join(res_folder_path,data_set_name_png))



if __name__ == '__main__':
    visulaize_tsne("input_df")
    visulaize_tsne("mask_x_df")
    visulaize_tsne("mask")