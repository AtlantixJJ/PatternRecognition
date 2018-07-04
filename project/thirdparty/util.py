#!/usr/bin/python3

import pandas as pd
import numpy as np
import os

keywords = [
    "lastPrice", "highestPrice", "lowestPrice",
    "volume", "turnover", "bidPrice1",
    "bidVolume1", "askPrice1", "askVolume1",
]

IDs = [ "A1", "A3", "B2", "B3" ]

def dataAlignment(files, IDs, keywords, interval=10, k=0.3):
    """Do data alignment for each files, shape a np array
       of len(IDs) * len(keywords) matrix, with certain interval

       Args:
           files (list(str)) filename list of csv file
           IDs   (list(str)) ID order of files's filename
           k     (float)     k value to calculate P05

       Returns:
            3 dimension np array with shape (times, IDs, keywords)
    """
    dfs = []
    for i in range(len(IDs)):
        df = pd.read_csv(files[i], index_col=0)
        print(files[i])
        df.index = pd.to_datetime(df.index, errors='coerce')
        df.dropna(inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        dfs.append(df)

    start_time = dfs[0].index[0]
    end_time   = dfs[0].index[-1]

    # TODO: "volumn" may need further handle
    current_time = start_time + pd.Timedelta(minutes=20)
    res = []
    while (current_time < end_time):
        values = []
        ids = [ 0 for _ in range(len(IDs)) ]
        for i in range(len(IDs)):
            try:
                value = dfs[i].loc[current_time.strftime("%Y-%m-%d %H:%M:%S")][keywords].values
                ids[i] = dfs[i].index.get_loc(current_time.strftime("%Y-%m-%d %H:%M:%S"))
            except KeyError:
                break
            if ( len(value.shape) == 2 and value.shape[0] > 0 ):
                value = value[0]
            elif ( len(value.shape) == 1 ):
                pass
            else:
                break
            values.append(value)
        if len(values) != len(IDs):
            current_time += pd.Timedelta(seconds=1)
            continue

        # calculate P05 for each ID
        P05s = []
        for i in range(len(IDs)):
            idx = ids[i]
            if type(idx) == slice:
                idx = idx.start
            elif type(idx) == np.ndarray:
                idx = idx[0]
            while (True):
                volumn1 = dfs[i].iloc[idx]['volume']
                volumn2 = dfs[i].iloc[max(idx-1, 0)]['volume']
                if ( volumn1 - volumn2 > 0 ):
                    turnover1 = dfs[i].iloc[idx]['turnover']
                    turnover2 = dfs[i].iloc[idx-1]['turnover']
                    bidPrice = dfs[i].iloc[idx]['bidPrice1']
                    askPrice = dfs[i].iloc[idx]['askPrice1']
                    P05s.append( k * (turnover1-turnover2)/float(volumn1-volumn2)
                                + (1-k) * (bidPrice+askPrice) / 2.)
                    break
                else:
                    idx = idx - 1
                    if (idx == 0):
                        bidPrice = dfs[i].iloc[idx+1]['bidPrice1']
                        askPrice = dfs[i].iloc[idx+1]['askPrice1']
                        P05s.append( (bidPrice + askPrice) / 2. )
        # change turnover into P05
        values = np.array(values)
        values[:, 4] = P05s
        res.append(values)
        current_time += pd.Timedelta(seconds=interval)
    res = np.array(res)

    return res


def createDatamat(data_path, IDs):
    filenames = [ f for f in os.listdir(data_path)
                 if (os.path.isfile(data_path+f) and f[-3:] == 'csv')]
    while ( len(filenames) != 0 ):
        date = filenames[0][2:-7]
        new_files = [ f for f in filenames if f[2:-7] == date ]
        ordered_files = []
        for ID in IDs:
            for f in new_files:
                if f[-6:-4] == ID:
                    ordered_files.append(data_path + f)
                    break
        result = dataAlignment(ordered_files, IDs, keywords, 5)
        # reshape the result matrix frome (times, IDs, keywords) to
        # (times*IDs, keywords) to save
        result = np.reshape(result, (-1, len(keywords)))
        np.savetxt(data_path + date + ".txt", result)
        filenames = list( filter(lambda x: x not in new_files, filenames) )
        print(ordered_files, " finished!")


# use for test only
if __name__ == "__main__":
    files = [ '0-20170703-day_A1.csv', '0-20170703-day_A3.csv',
              '1-20170703-day_B2.csv', '1-20170703-day_B3.csv' ]
    files = ['../data/' + filename for filename in files]
    result = dataAlignment(files, IDs, keywords, interval=10)
    print(result)
