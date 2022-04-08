import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix
import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self,main_folder_path = r"/media/data1/nivamitay/data/"  ,train_ratio=0.8,features=True):
        self.full_data = sc.read(os.path.join(main_folder_path,'10X_pbmc_5k_v3.h5ad'))
        self.colnames = self.full_data.var.index
        self.rownames = self.full_data.obs.index

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
        data[extra] = np.nan
        data = data[self.features]
        labels = self.full_data.obs[["cell_type_l1"]]
        labels = pd.get_dummies(labels["cell_type_l1"])

        x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=train_ratio,random_state=123)
        x_validation,x_test,y_validation,y_test = train_test_split(x_test,y_test,test_size=0.5,random_state=123)

        self.train_samples = x_train.index
        self.val_samples = x_validation.index
        self.test_samples = x_test.index
        
        
        self.x_train, self.y_train = x_train.values, y_train.values
        self.x_validation, self.y_validation = x_validation.values, y_validation.values
        self.x_test, self.y_test = x_test.values, y_test.values

        print(f"Load dataset: Total {self.labels.shape[0]} samples, {self.x_train.shape[1]} features\n\
            {len(labels.columns.values)} labels: {labels.columns.values}\n\
        X_train:{self.x_train.shape[0]} samples||X_val:{self.x_validation.shape[0]} samples||X_test:{self.x_test.shape[0]} samples")

        
        print(f"Train samples:{y_train.index}")
        print(f"Val samples:{y_validation.index}")
        print(f"Test samples:{y_test.index}")


