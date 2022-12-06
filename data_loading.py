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
from sklearn.utils.class_weight import compute_class_weight

import dask
import dask.dataframe as ddf
import pyarrow as pa
from collections import defaultdict



class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


class Data:
    def __init__(self, data_inst ,train_ratio=0.8,features=True,test_set=False,all_labels=False,feat_list = None):
        self.full_data = sc.read(data_inst.path)

        # filter_cell_types =['CD16+ NK','CD16- NK','Treg','classical monocyte','intermediate monocyte','memory B','memory CD4','memory CD8','myeloid DC','naive B','naive CD4','naive CD8','non-classical CD16+ monocyte','plasmacytoid DC']# None#["memory CD8","naive CD8"]#None#["memory CD8","naive CD8"]#None # "memory CD8"
        filter_cell_types = None#'undefined'
        labels_by = "cell_type_l2" 
        
        if filter_cell_types!= None:
            if isinstance(filter_cell_types,str):
                self.full_data = self.full_data[self.full_data.obs['cell_type_l2']==filter_cell_types,:]
            else:
                self.full_data = self.full_data[self.full_data.obs['cell_type_l2'].isin(filter_cell_types),:]

                
        self.colnames = self.full_data.var.index
        self.rownames = self.full_data.obs.index
        self.data_name = data_inst.name[:-5]

        if features:
            if feat_list is None:
                self.features = pd.read_csv('./data//narrow_subset_features.csv',index_col=0)['gene_name']
                # self.features = pd.read_csv('./data/our_features.csv',index_col=0)[self.data_name]
            else:
                self.features = feat_list
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
            data = pd.DataFrame(data.X.toarray())

        data.columns = self.overlap
        extra = list(np.setdiff1d(self.features,self.overlap))
        self.colnames = data.columns




        self.named_labels = self.full_data.obs[labels_by]#self.full_data.obs["subbatch"]#self.full_data.obs[cell_type_col]
        ##########################
        # delete = np.where(self.named_labels=="undefined")[0]
        # data = data.drop(delete)
        # delete_names = self.named_labels.index[delete]
        # self.named_labels = self.named_labels.drop(delete_names ,axis =0) 

        #################################################
        if "patient" in self.full_data.obs.columns:
            self.patient = self.full_data.obs["patient"]
        

        self.labels = pd.get_dummies(self.named_labels)
        if not all_labels:
            self.labels = self.labels[np.sort(np.unique(self.named_labels.values))]
        self.named_labels_uniq = np.array(self.labels.columns.categories)
        

        

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
            
            # logs = -torch.log(torch.from_numpy(y_train.values).float().mean(dim=0))
            # self.class_weights = (1/logs/(1/logs).sum())
            self.class_weights = torch.ones_like(torch.from_numpy(y_train.values).float().sum(dim=0))#1/torch.from_numpy(y_train.values)
            # self.class_weights = 1/torch.from_numpy(y_train.values).float().sum(dim=0)
        else:
            self.all_dataset   = ClassifierDataset(torch.from_numpy(data.values).float(), torch.from_numpy(self.labels.values).float())
        

        print(f"Loading {self.data_name} dataset:\n Total {self.labels.shape[0]} samples, {data.shape[1]} features\n\
            {len(self.named_labels_uniq)} labels: {self.named_labels_uniq}")
        
        
        if not test_set:
            print(f"X_train:{x_train.shape[0]} samples || X_val:{x_validation.shape[0]} samples || X_test:{x_test.shape[0]} samples")



class NymData:
    def __init__(self, data_inst ,train_ratio=0.8,features=False,test_set=False,all_labels=False):
        self.full_data = sc.read(data_inst.path)
        self.data_name = data_inst.name[:-5]

        filter_cell_types = None#["memory CD8","naive CD8"]#None#["memory CD8","naive CD8"]#None # "memory CD8"
        if self.data_name == "kang_2017_stim_pbmc":
            labels_by = "cell"
        elif self.data_name == "mouse_cortex_methods_comparison_log1p_cpm":
            labels_by = "CellType"
        else:
            labels_by = "cell_ontology_class"
        

        self.full_data = self.full_data[self.full_data.obs["age"]=="Y",:]
        # if filter_cell_types!= None:
        #     if isinstance(filter_cell_types,str):
        #         self.full_data = self.full_data[self.full_data.obs[labels_by]==filter_cell_types,:]
        #     else:
        #         self.full_data = self.full_data[self.full_data.obs[labels_by].isin(filter_cell_types),:]

                
        self.colnames = self.full_data.var.index
        self.rownames = self.full_data.obs.index
        

        if features:
            self.features = pd.read_csv('./data/scnym_data/features.csv',index_col=0)[self.data_name]
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
            data = pd.DataFrame(data.X.toarray())

        data.columns = self.overlap
        # extra = list(np.setdiff1d(self.features,self.overlap))
        self.colnames = data.columns


        self.named_labels = self.full_data.obs[labels_by]#self.full_data.obs["subbatch"]#self.full_data.obs[cell_type_col]
        if "patient" in self.full_data.obs.columns:
            self.patient = self.full_data.obs["patient"]
        

        self.labels = pd.get_dummies(self.named_labels)
        if not all_labels:
            self.labels = self.labels[np.sort(np.unique(self.named_labels.values))]
        self.named_labels_uniq = np.array(self.labels.columns.categories)

        

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
            
            # logs = -torch.log(torch.from_numpy(y_train.values).float().mean(dim=0))
            # self.class_weights = (1/logs/(1/logs).sum())
            self.class_weights = torch.ones_like(torch.from_numpy(y_train.values).float().sum(dim=0))#1/torch.from_numpy(y_train.values)
        else:
            self.all_dataset   = ClassifierDataset(torch.from_numpy(data.values).float(), torch.from_numpy(self.labels.values).float())
        
            print(f"Loading {self.data_name} dataset:\n Total {self.labels.shape[0]} samples, {data.shape[1]} features\n\
            {len(self.named_labels_uniq)} labels: {self.named_labels_uniq}")
        
        
        if not test_set:
            print(f"X_train:{x_train.shape[0]} samples || X_val:{x_validation.shape[0]} samples || X_test:{x_test.shape[0]} samples")