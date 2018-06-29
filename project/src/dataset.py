import os
import pandas as pd
import tensorflow as tf
import tensorpack as tp

def read_all_data(data_dir):
    files = os.listdir(data_dir)
    files.sort()
    
    dfs = []
    for f in files:
        if "csv" in f:
            print(f)
            dfs.append(pd.read_csv(os.path.join(data_dir, f), index_col=0, parse_dates=True))
    return dfs