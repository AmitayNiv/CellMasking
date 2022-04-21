import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

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

    feat_cols = data_set.columns[1:-2]

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)

    for i in range(int(data_set["y"].values.max())+1):
        current_data_set = data_set[data_set["y"]==float(i)]
        data_subset = current_data_set[feat_cols].values
    
        tsne_results = tsne.fit_transform(data_subset)
        current_data_set['tsne-2d-one'] = tsne_results[:,0]
        current_data_set['tsne-2d-two'] = tsne_results[:,1]

        plt.figure(figsize=(16,10))
        sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",
        palette=sns.color_palette("hls",len(np.unique(current_data_set.y.values))),
        data=current_data_set,
        legend="full",
        alpha=0.3).set(title=f"{data_set_name} | label:{i} |#samples:{current_data_set.shape[0]}")
        data_set_name_png =f"{i}_{data_set_name}_tSNE.png"
        plt.savefig(os.path.join(res_folder_path,"tsne",data_set_name_png))


def visulaize_imag(data_set_name):
    data_set_name_csv =data_set_name+".csv"
    res_folder_path = r'c:\Users\niv.a\Documents\GitHub\CellMasking\CellMasking\results'
    df_path = os.path.join(res_folder_path,data_set_name_csv)
    input_data = pd.read_csv(os.path.join(res_folder_path,'input_df.csv'),index_col=0)
    data_set = pd.read_csv(df_path,index_col=0)
    feat_cols = data_set.columns[1:-2]
    pca = PCA(n_components=2)
    for i in range(int(data_set["y"].values.max())+1):

        curr_input_data = input_data[input_data["y"]==float(i)]
        curr_input_sub_set =  curr_input_data[feat_cols].values

        current_data_set = data_set[data_set["y"]==float(i)]
        data_subset = current_data_set[feat_cols].values
    
        pca_result = pca.fit_transform(curr_input_sub_set)

        current_data_set['pca-one'] = pca_result[:,0]
        current_data_set['pca-two'] = pca_result[:,1] 

        current_data_set = current_data_set.sort_values(by='pca-one')

        plt.figure(figsize=(16,10))
        plt.imshow(current_data_set[feat_cols].values,cmap="hot")
        data_set_name_img =f"{i}_{data_set_name}_img.png"
        plt.savefig(os.path.join(res_folder_path,"img",data_set_name_img))


def visulaize_pca(data_set_name):
    data_set_name_csv =data_set_name+".csv"
    res_folder_path = r'c:\Users\niv.a\Documents\GitHub\CellMasking\CellMasking\results'
    df_path = os.path.join(res_folder_path,data_set_name_csv)

    data_set = pd.read_csv(df_path,index_col=0) 
    feat_cols = data_set.columns[1:-2]
    pca = PCA(n_components=2)
    for i in range(int(data_set["y"].values.max())+1):


        current_data_set = data_set[data_set["y"]==float(i)]
        data_subset = current_data_set[feat_cols].values
    
        pca_result = pca.fit_transform(data_subset)

        current_data_set['pca-one'] = pca_result[:,0]
        current_data_set['pca-two'] = pca_result[:,1] 


        plt.figure(figsize=(16,10))
        sns.scatterplot(
        x='pca-one', y="pca-two",
        hue="label",
        palette=sns.color_palette("hls",len(np.unique(current_data_set.y.values))),
        data=current_data_set,
        legend="full",
        alpha=0.3).set(title=f"{data_set_name} | label:{i} |#samples:{current_data_set.shape[0]}")
        data_set_name_png =f"{i}_{data_set_name}_PCA.png"
        plt.savefig(os.path.join(res_folder_path,"pca",data_set_name_png))


def visulaize_2d_var():
    res_folder_path = r'c:\Users\niv.a\Documents\GitHub\CellMasking\CellMasking\results'
    df_path_mask = os.path.join(res_folder_path,"mask.csv")
    mask_df = pd.read_csv(df_path_mask,index_col=0)
    
    feat_cols = mask_df.columns[1:-2]
    mask_vals = (mask_df[feat_cols].values>0.5).astype(int)


    df_path_input = os.path.join(res_folder_path,"input_df.csv")
    input_df = pd.read_csv(df_path_input,index_col=0)


    c =plt.cm.get_cmap('hsv',20)
    for i in range(int(input_df["y"].values.max())+1):
        current_mask_df = mask_df[mask_df["y"]==float(i)]
        current_mask_vals = (current_mask_df[feat_cols].values>0.5).astype(int)
        current_df = input_df[input_df["y"]==float(i)]
        input_vals = current_df[feat_cols].values

        bin_cropped_features = input_vals * current_mask_vals

        input_var = np.var(input_vals,axis=1)
        bin_cropped_var = np.var(bin_cropped_features,axis=1)

        f = plt.figure(i)
        
        plt.plot(input_var,bin_cropped_var,"o",color=c(i),label=str(i))
        plt.title(f"label:{i},#samples:{bin_cropped_var.shape[0]}")
        plt.savefig(r'c:\Users\niv.a\Documents\GitHub\CellMasking\CellMasking\results\var\{}.png'.format(i))


if __name__ == '__main__':
    visulaize_tsne("input_df")
    visulaize_tsne("mask_x_df")
    visulaize_tsne("mask")