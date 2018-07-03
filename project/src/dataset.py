import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorpack as tp

SLOPE_DENOTE_LEN = 40
TLEN = 40 # LEN of average time
THRESHOLD1 = 0.05 #0.002
THRESHOLD2 = 0.10 #0.004
INPUT_LEN = 200
OUTPUT_LEN = 40
DEP_LEN = OUTPUT_LEN + INPUT_LEN + 1

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
    def __init__(self, datasets, shuffle=True):
        self.shuffle = shuffle
        self.datasets = datasets

        self.len = min([len(d) for d in self.datasets])

    def size(self):
        return self.len

    def get_data(self):
        idxs = np.arange(self.len)
        if self.shuffle:
            self.rng.shuffle(idxs)
        
        for idx in idxs:
            yield [d[idx] for d in self.datasets]

class DataLoader(object):
    def __init__(self, dataset, batch_size, shuffle=True, num_threads=2):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataflow = RandomShuffler(dataset, self.shuffle)
        self.num_threads = num_threads
    
        self.ds1 = tp.dataflow.BatchData(self.dataflow, self.batch_size)
        self.ds2 = tp.dataflow.PrefetchData(self.ds1, nr_prefetch=32, nr_proc=self.num_threads)
        self.ds2.reset_state()

    def reset_state(self):
        tp.logger.info("Reset dataloader")
        self.dataflow.reset_state()
        self.ds1.reset_state()
        self.ds2.reset_state()

    def generator(self):
        return self.ds2.get_data()

class FuturesData(object):
    def __init__(self, debug=False, from_npz=False):
        self.data_dir = "futuresData"
        self.debug = debug

        if from_npz:
            self.__load_npz()
        else:
            self.__read_data()

        self.__proc()

    def __len__(self):
        return self.min_length

    def __save_npz(self):
        np.savez("futuresData", [self.all_data, self.earliest_time, self.length])

    def __load_npz(self):
        npz = np.load("futuresData.npz")['arr_0']
        self.all_data, self.earliest_time, self.length = npz

    def __read_data(self):
        files = os.listdir(self.data_dir)
        files.sort()
        
        self.all_data = [[], [], [], []]
        self.earliest_time = [[], [], [], []]
        self.length = [[], [], [], []]
        for f in files:
            if "csv" in f:
                df = pd.read_csv(os.path.join(self.data_dir, f), index_col=0, parse_dates=True)
                df_len = len(df[df.columns[0]])
                if df_len < 1000:
                    continue
                type_id = get_inst_type(f)
                df[u'meanPrice'] = get_mean_price(df[u'bidPrice1'])
                self.all_data[type_id].append(df[u'meanPrice'])
                self.earliest_time[type_id].append(df[df.columns[0]][INPUT_LEN+1])
                self.length[type_id].append(df_len)
        
        self.all_data = np.array(self.all_data)
        self.earliest_time = np.array(self.earliest_time)
        self.length = np.array(self.length)
    
    def __proc(self):
        self.file_number = self.all_data.shape[1]
        
        # generate sampling time
        self.min_length = self.length.min() - TLEN - 1
        self.idx = np.arange(self.min_length * self.file_number)

    def __getitem__(self, idx):
        file_idx = idx // self.min_length
        # retry until find a proper end time
        tried = []
        cnt = 0
        while True:
            cnt += 1
            if cnt > 4:
                # avoid rare error
                print("Cannot find!")
                idx = np.random.randint(self.min_length)
                tried = []

            train_seq = []
            target_seq = []
            # the inst_type used to mark an end
            end_type = np.random.randint(0, 4)
            if end_type in tried:
                continue
            else:
                tried.append(end_type)

            # use end inst type to find an ending at a timestamp
            scaled_idx = int((self.length[end_type, file_idx] - DEP_LEN) / float(self.min_length) * idx)
            barrier_time = self.all_data[end_type, file_idx].index[INPUT_LEN + scaled_idx]
            
            IS_FAILED = False
            end_idxs = []
            # exam other inst type
            for inst_type in range(4):
                # find the latest time before barrier
                end_idx = self.all_data[inst_type, file_idx].index.get_loc(barrier_time, 'ffill')
                end_idxs.append(end_idx)
                if end_idx < INPUT_LEN or end_idx + OUTPUT_LEN >= self.length[inst_type, file_idx]:
                    IS_FAILED = True
                    break
                train_seq.append(self.all_data[inst_type, file_idx][end_idx-INPUT_LEN:end_idx].as_matrix())
                target_seq.append(self.all_data[inst_type, file_idx][end_idx:end_idx+OUTPUT_LEN].as_matrix())

            # retry if failed
            if IS_FAILED:
                continue
            else:
                break

        # debug
        if self.debug:
            print("-----")
            for inst_type in range(4):
                if inst_type == end_type:
                    print(str(self.all_data[inst_type, file_idx].index[end_idxs[inst_type] - INPUT_LEN]) + " " + str(self.all_data[inst_type, file_idx].index[end_idxs[inst_type]]) + " " + str(self.all_data[inst_type, file_idx].index[end_idxs[inst_type] + OUTPUT_LEN]) + " *")
                else:
                    print(str(self.all_data[inst_type, file_idx].index[end_idxs[inst_type] - INPUT_LEN]) + " " + str(self.all_data[inst_type, file_idx].index[end_idxs[inst_type]]) + " " + str(self.all_data[inst_type, file_idx].index[end_idxs[inst_type] + OUTPUT_LEN]))
            print("-----")
        
        return np.array([train_seq, target_seq])
