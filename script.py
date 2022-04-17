import numpy as np
import pandas as pd
import dask
import dask.dataframe as ddf
import pyarrow as pa
import scipy as sp
from collections import defaultdict

from data_loading import ImmunData

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
        
    return meta, gex

if __name__ == '__main__':
    imm_data = ImmunData()
    meta, gex = imm_data.read_cells()
    # single_cell_dir = r'/media/data1/nivamitay/data/immunai/single-cell'
    # meta_cols = ['cell_id', 'project_id', 'sequencing_batch_id', 'lane_id', 'hashtag', 'test_set', 'tissue_type', 'cell_type']
    # meta, _ = read_cells(single_cell_dir,columns=None)
    # print(meta.shape)

    print()