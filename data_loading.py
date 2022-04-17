import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix
import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

import dask
import dask.dataframe as ddf
import pyarrow as pa
import scipy as sp


main_folder_path = r"/media/data1/nivamitay/data/" 

class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


class Data:
    def __init__(self, data_name ='10X_pbmc_5k_v3.h5ad' ,train_ratio=0.8,features=True,test_set=False):
        self.full_data = sc.read(os.path.join(main_folder_path,data_name))
        self.colnames = self.full_data.var.index
        self.rownames = self.full_data.obs.index
        self.data_name = data_name

        if features:
            self.features = pd.read_csv(os.path.join(main_folder_path,'features.csv'),index_col=0)['0']
            
        else:
            self.features = self.colnames

        self.overlap = list(set(self.colnames) & set(self.features))
        if len(self.overlap) < 100:
            sys.exit('Error: Not enough feature overlap.')

        if (type(self.full_data.X) == csc_matrix) | (type(self.full_data.X) == csr_matrix):
            data = self.full_data[:,self.overlap]
            data = pd.DataFrame.sparse.from_spmatrix(data.X)
        else:
            data = self.full_data[:,self.overlap]
            data = pd.DataFrame(data.X)

        data.columns = self.overlap
        data.index = self.rownames
        extra = list(np.setdiff1d(self.features,self.overlap))
        # data[extra] = np.nan
        # data = data[self.features]
        self.colnames = data.columns

        cell_type_col = "cell_type_l2"
        labels = self.full_data.obs[cell_type_col]
        self.labels = pd.get_dummies(labels)[np.unique(labels.values)]

        if not test_set:
            x_train,x_test,y_train,y_test = train_test_split(data,self.labels,test_size=(1-train_ratio),random_state=123)
            x_validation,x_test,y_validation,y_test = train_test_split(x_test,y_test,test_size=0.5,random_state=123)

            self.train_samples = x_train.index
            self.val_samples = x_validation.index
            self.test_samples = x_test.index
            
            self.train_dataset  = ClassifierDataset(torch.from_numpy(x_train.values).float(), torch.from_numpy(y_train.values).float())
            self.val_dataset   = ClassifierDataset(torch.from_numpy(x_validation.values).float(), torch.from_numpy(y_validation.values).float())
            self.test_dataset   = ClassifierDataset(torch.from_numpy(x_test.values).float(), torch.from_numpy(y_test.values).float())

            self.class_weights = 1./(torch.from_numpy(y_train.values).float().sum(dim=0))
        else:
            self.test_dataset   = ClassifierDataset(torch.from_numpy(data.values).float(), torch.from_numpy(self.labels.values).float())
        

        print(f"Loading {data_name} dataset:\n Total {self.labels.shape[0]} samples, {data.shape[1]} features\n\
            {len(np.unique(labels.values))} labels: { np.unique(labels.values)}")
        
        
        if not test_set:
            print(f"X_train:{x_train.shape[0]} samples || X_val:{x_validation.shape[0]} samples || X_test:{x_test.shape[0]} samples")
            print(f"Class weights: {self.class_weights}")

        
        # print(f"Train samples:{y_train.index}")
        # print(f"Val samples:{y_validation.index}")
        # print(f"Test samples:{y_test.index}")
       

class ImmunData:
    def __init__(self):

    def decode_csr_array(val):
        return sp.sparse.csr_matrix((val['data'], val['indices'], val['indptr']), shape=val['shape'])

def read_cells(single_cell_dir, columns=None, filters=None):
    result = ddf.read_parquet(
        path=single_cell_dir, 
        engine='pyarrow',
        columns=columns,
        filters=filters, 
        metadata_task_size=32, 
        split_row_groups=False
    ).compute()
    
    meta = result
    gex = None
    if (columns and 'gex' in columns) or columns is None:
        gex = sp.sparse.vstack([decode_csr_array(cell_gex) for cell_gex in result['gex']])
        meta = result.drop('gex', axis=1)
            
