import scanpy as sc
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix
import sys
import os
import numpy as np
pmbc_rna = sc.read_h5ad(r"/media/data1/nivamitay/data/10X_pbmc_5k_v3.h5ad")
pmbc_rna_adt = sc.read_h5ad(r"/media/data1/nivamitay/data/10X_pbmc_5k_v3-adt.h5ad")



features = pd.read_csv(r"/media/data1/nivamitay/data/features.csv", index_col=0)['0']


class Data:
    def __init__(self,main_folder_path = r"/media/data1/nivamitay/data/"  ,train_ratio=0.8):
        self.features = pd.read_csv(os.path.join(main_folder_path,'features.csv'),index_col=0)['0']
        self.full_data = sc.read(os.path.join(main_folder_path,'10X_pbmc_5k_v3.h5ad'))
        self.colnames = data.var.index
        self.rownames = data.obs.index
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
        extra = list(np.setdiff1d(features,self.overlap))
        data[extra] = np.nan
        self.data = data[features].values
        labels = pmbc_rna.obs[["cell_type_l1"]]
        self.labels = pd.get_dummies(labels["cell_type_l1"]).values
