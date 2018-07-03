import os
import pandas as pd
import tensorflow as tf
import tensorpack as tp

SLOPE_DENOTE_LEN = 40
TLEN = 40 # LEN of average time
THRESHOLD1 = 0.05 #0.002
THRESHOLD2 = 0.10 #0.004

def get_class(div):
    """
    Classify classes of increase or decrease, by slope
    """
    ans = -1
    MID = 0

    if div > MID + THRESHOLD2:
        ans = 5
    elif div > MID + THRESHOLD1:
        ans = 4
    elif div > MID - THRESHOLD1:
        ans = 3
    elif div > MID - THRESHOLD2:
        ans = 2
    else:
        ans = 1

    return ans

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

def label_slope(slope):
    """
    Give label to sequence
    """
    return np.array([get_class(slope[i]) for i in range(slope.shape[0]-TLEN)])

def get_mean_price(bid_price):
    """
    Args:
    bid_price:  A dataframe
    """

    mean_price = bid_price.rolling(window=TLEN, min_periods=1).mean()
    return mean_price

def get_inst_type(name):
    INST_TYPE = ["A1", "A3", "B2", "B3"]
    for i in range(len(INST_TYPE)):
        if INST_TYPE[i] in name:
            return i


class RandomShuffler(tp.dataflow.RNGDataFlow):
    def __init__(self, datasets):
        if type(datasets) == list and len(datasets) == 1:
            self.datasets = datasets[0]
        else:
            self.datasets = datasets

    def get_data(self):
        idxs = np.arange(len(self.files))
        if self.shuffle:
            self.rng.shuffle(idxs)
    

class FuturesData(object):
    def __init__(self, data_dir, shuffle=True):
        self.data_dir = data_dir
        self.shuffle = shuffle
    
        self.__read_data()

    def __read_data(self):
        files = os.listdir(self.data_dir)
        files.sort()
        
        self.all_data = [[], [], [], []]
        for f in files:
            if "csv" in f:
                print(f)
                df = pd.read_csv(os.path.join(self.data_dir, f), index_col=0, parse_dates=True)
                type_id = get_inst_type(f)
                df[u'meanPrice'] = get_mean_price(df[u'bidPrice1'])
                self.all_data[type_id].append(df[u'meanPrice'])
        
        # generate sampling time

    
    def __getitem__(self, idx):
        pass
                
