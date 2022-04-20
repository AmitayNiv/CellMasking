import scanpy as sc
import pandas as pd
import scipy as sp
from scipy.sparse import csr_matrix, csc_matrix
import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
import torch

import dask
import dask.dataframe as ddf
import pyarrow as pa
from collections import defaultdict





main_folder_path = r"/media/data1/nivamitay/CellMasking/data/" 

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
        self.overlap.sort()
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
        self.named_labels = self.full_data.obs[cell_type_col]
        self.labels = pd.get_dummies(self.named_labels)[np.unique(self.named_labels.values)]

        self.number_of_classes = self.labels.shape[1]
        self.n_features = len(self.colnames)

        if not test_set:
            x_train,x_test,y_train,y_test = train_test_split(data,self.labels,test_size=(1-train_ratio),random_state=None)
            x_validation,x_test,y_validation,y_test = train_test_split(x_test,y_test,test_size=0.5,random_state=None)

            self.train_samples = x_train.index
            self.val_samples = x_validation.index
            self.test_samples = x_test.index
            
            self.train_dataset  = ClassifierDataset(torch.from_numpy(x_train.values).float(), torch.from_numpy(y_train.values).float())
            self.val_dataset   = ClassifierDataset(torch.from_numpy(x_validation.values).float(), torch.from_numpy(y_validation.values).float())
            self.test_dataset   = ClassifierDataset(torch.from_numpy(x_test.values).float(), torch.from_numpy(y_test.values).float())

            self.all_dataset = ClassifierDataset(torch.from_numpy(data.values).float(), torch.from_numpy(self.labels.values).float())
            
            self.class_weights = 1./(torch.from_numpy(y_train.values).float().sum(dim=0))
        else:
            self.test_dataset   = ClassifierDataset(torch.from_numpy(data.values).float(), torch.from_numpy(self.labels.values).float())
        

        print(f"Loading {data_name} dataset:\n Total {self.labels.shape[0]} samples, {data.shape[1]} features\n\
            {len(np.unique(self.named_labels.values))} labels: { np.unique(self.named_labels.values)}")
        
        
        if not test_set:
            print(f"X_train:{x_train.shape[0]} samples || X_val:{x_validation.shape[0]} samples || X_test:{x_test.shape[0]} samples")
            print(f"Class weights: {self.class_weights}")


        
        # print(f"Train samples:{y_train.index}")
        # print(f"Val samples:{y_validation.index}")
        # print(f"Test samples:{y_test.index}")
       

class ImmunData:
    def __init__(self,data_set = None,genes_filter="all",test_set=False,all_types = False):
        self.single_cell_dir = os.path.join(main_folder_path,'immunai/single-cell')
        self.cell_type_hierarchy = pd.read_csv(os.path.join(main_folder_path,'immunai/cell_types.csv'))
        self.genes = pd.read_csv(os.path.join(main_folder_path,'immunai/genes.csv.gz'), index_col='gene_index')
        self.data_set = data_set
        self.all_types = all_types
        self.meta_cols = ['cell_id', 'project_id', 'sequencing_batch_id', 'lane_id', 'hashtag', 'test_set', 'tissue_type', 'cell_type']

        self.init_data_types()

        filters = [] if data_set==None else [('tissue_type', '=', data_set)]
        train_filters = filters.copy()
        train_filters.append(("test_set","=",False))
        test_filters = filters.copy()
        test_filters.append(("test_set","=",True))


        train_gex, train_meta = self.read_cells(filters=train_filters)
        test_gex, test_meta = self.read_cells(filters=test_filters)

        if genes_filter!="all":
            test_gex = test_gex[:,self.genes[genes_filter]]
            train_gex = train_gex[:,self.genes[genes_filter]]

        train_labels = np.zeros((train_meta["int_cell_type"].shape[0],self.number_of_classes))
        train_labels[np.arange(train_meta["int_cell_type"].shape[0]),train_meta["int_cell_type"]] = 1

        test_labels = np.zeros((test_meta["int_cell_type"].shape[0],self.number_of_classes))
        test_labels[np.arange(test_meta["int_cell_type"].shape[0]),test_meta["int_cell_type"]] = 1

        x_validation,x_test,y_validation,y_test = train_test_split(test_gex,test_labels,test_size=0.5,random_state=123)

        self.train_dataset  = ClassifierDataset(torch.from_numpy(train_gex.todense()).float(), torch.from_numpy(train_labels).float())
        self.val_dataset   = ClassifierDataset(torch.from_numpy(x_validation.todense()).float(), torch.from_numpy(y_validation).float())
        self.test_dataset   = ClassifierDataset(torch.from_numpy(x_test.todense()).float(), torch.from_numpy(y_test).float())

        active_classes = len(np.unique(train_meta["int_cell_type"].values))
        if not all_types:
            self.class_weights = 1./(torch.from_numpy(train_labels).float().sum(dim=0))


        self.n_features = test_gex.shape[1]
        name = "All" if data_set==None else data_set
        print(f"Loading {name} dataset:\n Total {test_meta.shape[0]+train_meta.shape[0]} samples, {self.n_features } features\n\
            {active_classes}/{self.number_of_classes} labels")

        if not test_set:
            print(f"X_train:{train_gex.shape[0]} samples || X_val:{x_validation.shape[0]} samples || X_test:{x_test.shape[0]} samples")
            print(f"Class weights:{self.class_weights}")


    def init_data_types(self):
        filters = [] if self.data_set==None else [('tissue_type', '=', self.data_set)]
        _, self.meta = self.read_cells(columns=self.meta_cols,filters = filters)
        if self.all_types:
            self.cell_types = self.cell_type_hierarchy['cell_type']
            self.named_labels = self.cell_types.values
            self.map = dict(zip(self.cell_types.values, self.cell_types.index))
            self.number_of_classes = self.cell_types.shape[0]
            self.get_cell_type_weights()
        else:
            self.existing_types = np.unique(self.meta["cell_type"].values)
            self.named_labels = self.existing_types 
            self.map = dict(zip(self.existing_types, list(range(self.existing_types.shape[0]))))
            self.number_of_classes = self.existing_types.shape[0]
        


    def get_cell_type_weights(self):
        def get_cell_type_descendants(cell_type):
            descendants = cell_type_children[cell_type]
            for child in descendants:
                descendants = descendants | get_cell_type_descendants(child)
                
            return descendants

        n_cell_types = self.cell_type_hierarchy.shape[0]
        self.cell_types = self.cell_type_hierarchy['cell_type']
        cell_type_parent = dict(zip(self.cell_types, self.cell_type_hierarchy['parent_cell_type']))
        cell_type_indices = dict(zip(self.cell_types, self.cell_type_hierarchy.index.values))

        cell_type_children = defaultdict(set)
        for child_cell_type, parent_cell_type in cell_type_parent.items():
            cell_type_children[parent_cell_type].add(child_cell_type)

        

        ## each cell type represented by a row, with descendents of that cell type indicated as ones in columns
        cell_type_descendents = np.zeros((n_cell_types, n_cell_types), dtype=int)
        for parent_cell_type, parent_cell_type_idx in cell_type_indices.items():
            cell_type_descendents[parent_cell_type_idx, parent_cell_type_idx] = 1
            for descendent_cell_type in get_cell_type_descendants(parent_cell_type):
                descendent_cell_type_idx = cell_type_indices[descendent_cell_type]
                cell_type_descendents[parent_cell_type_idx, descendent_cell_type_idx] = 1

        annotated_cell_types = np.zeros((self.meta.shape[0], len(self.cell_types)), dtype=int)
        for cell_idx, annotated_cell_type in enumerate(self.meta['cell_type']):
            annotated_cell_types[cell_idx, cell_type_indices[annotated_cell_type]] = 1
        annotated_cell_type_ancestors = np.matmul(annotated_cell_types, cell_type_descendents.transpose())
        cell_type_abundances = pd.DataFrame(data={
            'cell_type': self.cell_types, 
            'abundance': np.mean(annotated_cell_type_ancestors, axis=0)
        }).sort_values(by='abundance', ascending=False)
        cell_type_abundances = cell_type_abundances.sort_index()
        # cell_type_abundances = cell_type_abundances[cell_type_abundances['abundance'] > 0]
        cell_type_abundances['neg-log-abundance'] = -np.log(cell_type_abundances['abundance'])
        cell_type_abundances['class_weight'] = 1 / cell_type_abundances['neg-log-abundance'] / np.sum(1 / cell_type_abundances['neg-log-abundance'])
        
        self.class_weights = cell_type_abundances['class_weight'].values 

    def convert_map(self,x):
        return self.map[x]

    def decode_csr_array(self,val):
        return sp.sparse.csr_array((val['data'], val['indices'], val['indptr']), shape=val['shape'])
        
    def read_cells(self,columns=None, filters=None):
        result = ddf.read_parquet(
            path=self.single_cell_dir, 
            engine='pyarrow',
            columns=columns,
            filters=filters, 
            metadata_task_size=32, 
            split_row_groups=False
        ).compute()
        
        meta = result
        gex = None
        if (columns and 'gex' in columns) or columns is None:
            gex = sp.sparse.vstack([self.decode_csr_array(cell_gex) for cell_gex in result['gex']])
            meta  = result.drop('gex', axis=1)
            meta["int_cell_type"] = meta["cell_type"].apply(self.convert_map)
        return gex, meta
        
        # self.train_lanes = set(self.meta[~self.meta['test_set']]['lane_id'].unique())
        # self.test_lanes = set(self.meta[self.meta['test_set']]['lane_id'].unique())

        
