import os
import pandas as pd
import tensorflow as tf
import tensorpack as tp

SLOPE_DENOTE_LEN = 40
TLEN = 40 # LEN of average time

def get_fit_slope(mean_price):
    """
    Compute slope in future 40 points in order to determine its tendency.
    """
    x = np.array(range(0, SLOPE_DENOTE_LEN))
    x2 = x ** 2
    sum_x = x.sum()
    sum_x2 = x2.sum()
    div = SLOPE_DENOTE_LEN * sum_x2 - sum_x ** 2

    Y = mean_price
    k = np.zeros_like(Y, dtype="float32")
    for i in range(Y.shape[0]-SLOPE_DENOTE_LEN):
        if Y[i] < 1:
            break
        y = Y[i:i+SLOPE_DENOTE_LEN]
        k[i] = ((SLOPE_DENOTE_LEN * (x * y).sum() - sum_x * y.sum()) / div) * 5000.0 / Y[i]

    return k

def get_mean_price(bid_price):
    """
    Args:
    bid_price:  A dataframe
    """

    mean_price = bid_price.rolling(window=40, min_periods=1).mean()
    return mean_price

def read_all_data(data_dir):
    files = os.listdir(data_dir)
    files.sort()
    
    dfs = []
    for f in files:
        if "csv" in f:
            print(f)
            dfs.append(pd.read_csv(os.path.join(data_dir, f), index_col=0, parse_dates=True))
    return dfs