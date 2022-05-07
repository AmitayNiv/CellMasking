import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pkg_resources import safe_name
import umap


def visulaize_tsne(data_set,table_name,data_name,wandb_exp=None):
    # data_set_name_csv =data_set_name+".csv"
    # res_folder_path = r'c:\Users\niv.a\Documents\GitHub\CellMasking\CellMasking\results'
    # df_path = os.path.join(res_folder_path,data_set_name_csv)
    # data_set = pd.read_csv(df_path,index_col=0)

    feat_cols = data_set.columns[1:-2]

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)

    # for i in range(int(data_set["y"].values.max())+1):
    current_data_set = data_set#[data_set["y"]==float(i)]
    data_subset = current_data_set[feat_cols].values

    tsne_results = tsne.fit_transform(data_subset)
    current_data_set['tsne-2d-one'] = tsne_results[:,0]
    current_data_set['tsne-2d-two'] = tsne_results[:,1]


    fig, ax = plt.subplots(figsize=(16,10))
    sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="label",
    palette=sns.color_palette("hls",15),
    data=current_data_set,
    legend="full",
    alpha=0.3).set(title=f"{table_name}|{data_name}|#samples:{current_data_set.shape[0]}")
    data_set_name_png =f"tsne_{table_name}.png"
    res_folder_path = f"./results/{data_name}/"
    plt.savefig(os.path.join(res_folder_path,data_set_name_png))

        # plt.savefig(os.path.join(res_folder_path,r"plots\tsne",data_set_name_png))
        # wandb_exp.log({f"{table_name} | label:{i} |#samples:{current_data_set.shape[0]}":fig})
    plt.cla()
    plt.close("all")

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
        plt.savefig(os.path.join(res_folder_path,r"plots\img",data_set_name_img))
    plt.cla()
    plt.close("all")

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
        palette=sns.color_palette("hls",len(np.unique(current_data_set["label"].values))),
        data=current_data_set,
        legend="full",
        alpha=0.3).set(title=f"{data_set_name} | label:{i} |#samples:{current_data_set.shape[0]}")
        data_set_name_png =f"{i}_{data_set_name}_PCA.png"
        plt.savefig(os.path.join(res_folder_path,r"plots\pca",data_set_name_png))
    plt.cla()
    plt.close("all")

def visulaize_2d_var():
    res_folder_path = r'c:\Users\niv.a\Documents\GitHub\CellMasking\CellMasking\results'
    df_path_mask = os.path.join(res_folder_path,"mask.csv")
    mask_df = pd.read_csv(df_path_mask,index_col=0)
    
    feat_cols = mask_df.columns[1:-2]

    df_path_input = os.path.join(res_folder_path,"input_df.csv")
    input_df = pd.read_csv(df_path_input,index_col=0)


    c =plt.cm.get_cmap('hsv',20)
    for i in range(int(input_df["y"].values.max())+1):
        current_mask_df = mask_df[mask_df["y"]==float(i)]
        current_mask_vals = (current_mask_df[feat_cols].values>0.5).astype(int)
        current_df = input_df[input_df["y"]==float(i)]
        input_vals = current_df[feat_cols].values

        # bin_cropped_features = input_vals * current_mask_vals


        samp_idx,gene_idx = np.where(current_mask_vals>0)

        # var = []
        # for idx in gene_idx:
        #     bin_cropped_features = input_vals[samp_idx,gene_idx]

        var = []
        for gene_idx in range(input_vals.shape[1]):
            gene_vec = []
            for sample_idx in range(input_vals.shape[0]):
                if current_mask_vals[sample_idx,gene_idx]>0:
                    gene_vec.append(input_vals[sample_idx,gene_idx])
            var.append(np.var(gene_vec))
        



        input_var = np.var(input_vals,axis=0)
        # bin_cropped_var = np.var(bin_cropped_features)

        f = plt.figure(i)
        
        plt.plot(input_var,var,"o",color=c(i),label=str(i))
        plt.title(f"label:{i},#samples:{input_vals.shape[0]}")
        plt.savefig(r'c:\Users\niv.a\Documents\GitHub\CellMasking\CellMasking\results\plots\var\{}.png'.format(i))
    plt.cla()
    plt.close("all")


def visulaize_umap(data_set,table_name,data_name):
    feat_cols = data_set.columns[1:-2]
    reducer = umap.UMAP(random_state=42)
    
    current_data_set = data_set.loc[(data_set['label']=="memory CD8") | (data_set['label']=="naive CD8")]
    data_subset = current_data_set[feat_cols].values

    reducer.fit(data_subset)
    embedding = reducer.transform(data_subset)
    
    current_data_set['embedding_0'] = embedding[:,0]
    current_data_set['embedding_1'] = embedding[:,1]


    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x='embedding_0', y='embedding_1',
        hue="label",
        palette=sns.color_palette("hls",15),
        data=current_data_set,
        legend="full",
        alpha=0.3)
    plt.title(f'UMAP projection of the {table_name}| dataset:{data_name}', fontsize=16)


    data_set_name_png =f"umap_{table_name}.png"
    res_folder_path = f"./results/{data_name}/"
    plt.savefig(os.path.join(res_folder_path,data_set_name_png))

    plt.cla()
    plt.close("all")


if __name__ == '__main__':
    visulaize_tsne("mask")
    visulaize_tsne("input_df")
    visulaize_tsne("mask_x_df")
    visulaize_tsne("mask_inv")

    

    # visulaize_imag("input_l1")
    # visulaize_imag("mask_x_l1")
    # visulaize_imag("mask_l1")

    visulaize_pca("input_df")
    visulaize_pca("mask_x_df")
    visulaize_pca("mask")
    visulaize_pca("mask_inv")

    # visulaize_2d_var()