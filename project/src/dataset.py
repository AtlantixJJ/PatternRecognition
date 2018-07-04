import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorpack as tp

THRESHOLD1 = 0.005 #0.001
THRESHOLD2 = 0.02 #0.002
TLEN = 100 # LEN of average time
SLOPE_DENOTE_LEN = 40
INPUT_LEN = 200
OUTPUT_LEN = 40
DEP_LEN = OUTPUT_LEN + INPUT_LEN + 1

tmp_x = np.array(range(0, SLOPE_DENOTE_LEN))
tmp_x2 = tmp_x ** 2
sum_tmp_x = tmp_x.sum()
sum_tmp_x2 = tmp_x2.sum()
tmp_div = SLOPE_DENOTE_LEN * sum_tmp_x2 - sum_tmp_x ** 2

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

    mean_price = bid_price.copy(True)
    for i in range(TLEN//2, mean_price.shape[0]-TLEN//2):
        mean_price[i] = bid_price[i-TLEN//2:i+TLEN//2].mean()
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
        for i, idx in enumerate(idxs):
            yield [d[idx] for d in self.datasets]

class FuturesData(object):
    def __init__(self, is_train=True, debug=False, from_npz=True):
        self.data_dir = "futuresData"
        self.debug = debug
        self.is_train = is_train

        self.train_val_splitor = pd.Timestamp(year=2017, month=8, day=10)

        if from_npz:
            self.__load_npz()
        else:
            self.__read_data()
            self.__save_npz()

        self.__proc()

    def __len__(self):
        if self.is_train:
            return self.train_number
        else:
            return self.test_number

    def __save_npz(self):
        #np.savez("futuresData", [self.all_data, self.earliest_time, self.length])
        np.savez("futures_train_val",
            [self.train_all_data, self.train_earliest_time, self.train_length],
            [self.test_all_data, self.test_earliest_time,   self.test_length])

    def __load_npz(self):
        print("Load from futures_train_val.npz")
        res = np.load("futures_train_val.npz")
        self.train_all_data, self.train_earliest_time, self.train_length = res['arr_0']
        self.test_all_data, self.test_earliest_time,   self.test_length  = res['arr_1']
        #self.all_data, self.earliest_time, self.length = npz
        print("Done")

    def __read_data(self):
        """
        Read data and divide 
        """
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
                x = get_mean_price(df[u'bidPrice1'])
                x = x[~x.index.duplicated(keep='first')]
                self.all_data[type_id].append(x)
                self.all_data[type_id][-1].sort_index(inplace=True)
                self.earliest_time[type_id].append(pd.Timestamp(df.index[INPUT_LEN+1]))
                self.length[type_id].append(df_len)
        
        self.all_data = np.array(self.all_data)
        self.earliest_time = np.array(self.earliest_time)
        self.length = np.array(self.length)

        self.file_number = self.all_data.shape[1]
        
        # generate sampling time
        self.min_length = self.length.min() - TLEN - 1
        self.idx = np.arange(self.min_length * self.file_number)

        # divide train/test split
        self.train_all_data = [[], [], [], []]
        self.train_earliest_time = [[], [], [], []]
        self.train_length = [[], [], [], []]

        self.test_all_data = [[], [], [], []]
        self.test_earliest_time = [[], [], [], []]
        self.test_length = [[], [], [], []]

        for inst_type in range(4):
            for i in range(self.all_data.shape[1]):
                if self.earliest_time[inst_type, i] < self.train_val_splitor:
                    self.train_all_data[inst_type].append(self.all_data[inst_type, i])
                    self.train_earliest_time[inst_type].append(self.earliest_time[inst_type, i])
                    self.train_length[inst_type].append(self.length[inst_type, i])
                else:
                    self.test_all_data[inst_type].append(self.all_data[inst_type, i])
                    self.test_earliest_time[inst_type].append(self.earliest_time[inst_type, i])
                    self.test_length[inst_type].append(self.length[inst_type, i])

        self.train_all_data         = np.array(self.train_all_data         )
        self.train_earliest_time    = np.array(self.train_earliest_time    )
        self.train_length           = np.array(self.train_length           )
        self.test_all_data          = np.array(self.test_all_data          )
        self.test_earliest_time     = np.array(self.test_earliest_time     )
        self.test_length            = np.array(self.test_length            )

    def __proc(self):
        """
        Collect some extra information
        """
        self.train_file_number = self.train_all_data.shape[1]
        self.train_min_length = self.train_length.min() - DEP_LEN
        self.train_number = self.train_min_length * self.train_file_number

        self.test_file_number = self.test_all_data.shape[1]
        self.test_min_length = self.test_length.min() - DEP_LEN
        self.test_number = self.test_min_length * self.test_file_number

    def __getitem__(self, idx):
        if self.is_train:
            number          = self.train_number
            min_length      = self.train_min_length
            length          = self.train_length
            file_number     = self.train_file_number
            all_data        = self.train_all_data
        else:
            number          = self.test_number
            min_length      = self.test_min_length
            length          = self.test_length
            file_number     = self.test_file_number
            all_data        = self.test_all_data

        file_idx = idx // min_length
        idx = idx % min_length
    
        # retry until find a proper end time
        tried = []
        while True:
            if len(tried) == 4:
                # avoid rare error
                tp.logger.info("Cannot find!")
                idx = np.random.randint(number)
                tried = []

                file_idx = idx // min_length
                idx = idx % min_length

            train_seq = []
            target_seq = []
            k = []
            # the inst_type used to mark an end
            
            while True:
                end_type = np.random.randint(0, 4)
                if end_type not in tried:
                    break
                
            tried.append(end_type)
            
            if idx >= min_length:
                tp.logger.info("Fatal error: %d %d " % (idx, min_length))

            # use end inst type to find an ending at a timestamp
            scaled_idx = int((length[end_type, file_idx] - DEP_LEN) * idx / float(min_length))
            barrier_time = all_data[end_type, file_idx].index[INPUT_LEN + scaled_idx]
            
            IS_FAILED = False
            end_idxs = []
            # exam other inst type
            for inst_type in range(4):
                # find the latest time before barrier
                end_idx = all_data[inst_type, file_idx].index.get_loc(str(barrier_time), 'ffill')
                try:
                    end_idx = int(end_idx)
                except TypeError:
                    print(end_idx)
                    end_idx = end_idx.start
                end_idxs.append(end_idx)
                if end_idx < INPUT_LEN or end_idx + OUTPUT_LEN >= length[inst_type, file_idx]:
                    IS_FAILED = True
                    break

            # retry if failed
            if IS_FAILED:
                continue

            for i, end_idx in enumerate(end_idxs):
                train_seq.append(all_data[i, file_idx].values[end_idx-INPUT_LEN:end_idx])
                target_seq.append(all_data[i, file_idx].values[end_idx:end_idx+OUTPUT_LEN])
                slope = (OUTPUT_LEN * (tmp_x * target_seq[i]).sum() - sum_tmp_x * target_seq[i].sum()) / tmp_div / 5000.0
                k.append(get_class(slope))
            break

        # debug
        if self.debug:
            print("-----")
            for inst_type in range(4):
                if inst_type == end_type:
                    print(str(all_data[inst_type, file_idx].index[end_idxs[inst_type] - INPUT_LEN]) + " " + str(all_data[inst_type, file_idx].index[end_idxs[inst_type]]) + " " + str(all_data[inst_type, file_idx].index[end_idxs[inst_type] + OUTPUT_LEN]) + " *")
                else:
                    print(str(all_data[inst_type, file_idx].index[end_idxs[inst_type] - INPUT_LEN]) + " " + str(all_data[inst_type, file_idx].index[end_idxs[inst_type]]) + " " + str(all_data[inst_type, file_idx].index[end_idxs[inst_type] + OUTPUT_LEN]))
            print("-----")
        
        return {'seq': np.array(train_seq),
                'target': np.array(target_seq),
                'label' : np.array(k)}

### Fucking deprecated ###

class DataLoader(object):
    def __init__(self, datasets, batch_size, shuffle=True, num_threads=2):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataflow = RandomShuffler(datasets, self.shuffle)
        self.num_threads = num_threads
    
        self.ds1 = tp.dataflow.BatchData(self.dataflow, self.batch_size, use_list=True)
        self.ds2 = tp.dataflow.MultiProcessMapDataZMQ(self.ds1, nr_prefetch=1, nr_proc=self.num_threads)
        self.ds2.reset_state()

    def reset_state(self):
        #tp.logger.info("Reset dataloader")
        self.dataflow.reset_state()
        self.ds1.reset_state()
        self.ds2.reset_state()

    def generator(self):
        return self.ds2.get_data()