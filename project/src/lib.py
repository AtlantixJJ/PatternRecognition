import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.plot(range(10))
plt.close()

import pandas as pd
import numpy as np

INST_TYPE = ["A1", "A3", "B2", "B3"]
THRESHOLD1 = 0.005 #0.001
THRESHOLD2 = 0.02 #0.002
TLEN = 100 # LEN of average time
SLOPE_DENOTE_LEN = 40
MEANPRICE = "meanPrice"+str(TLEN)
SLOPEPRICE = "slopePrice"+str(SLOPE_DENOTE_LEN)
CLASSIFICATION_METHOD = "slope"


### process ###
###############

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

def get_class_maxdiff(pd, nd):
    """
    pd, nd: positive difference, negative difference
    """
    THETA_1 = 0.1 / 100
    THETA_2 = 0.2 / 100

    if pd > nd:
        if pd > THETA_2:
            return 5
        elif pd > THETA_1:
            return 4
        elif pd >= 0:
            return 3
        else:
            print("Wrong maxdiff: positive difference is negative")
    else:
        if nd > THETA_2:
            return 1
        elif nd > THETA_1:
            return 2
        elif nd >= 0:
            return 3
        else:
            print("Wrong maxdiff: negative difference is negative")


def get_mean_price(dic):
    """
    Smooth the data by averaging with window size 200.
    """
    for inst_type in INST_TYPE:
        try:
            dic[inst_type]['bidPrice1']
        except KeyError:
            #print(inst_type + " not exist")
            continue

        x = pd.DataFrame(dic[inst_type]['bidPrice1'])
        m = np.zeros_like(x)
        m[:TLEN] = x[:TLEN]
        m[-TLEN:] = x[-TLEN:]
        for i in range(TLEN//2, x.shape[0]-TLEN//2):
            m[i] = x[i-TLEN//2:i+TLEN//2].mean()
        dic[inst_type][MEANPRICE] = m
        #dic[inst_type][MEANPRICE] = x.rolling(window=TLEN, min_periods=1, center=True).mean().as_matrix()[:, 0]
        print(dic[inst_type][MEANPRICE].shape)
        print(dic[inst_type][MEANPRICE][:10])

def get_fit_slope(dic):
    """
    Compute slope in future 40 points in order to determine its tendency.
    """
    x = np.array(range(0, SLOPE_DENOTE_LEN))
    x2 = x ** 2
    sum_x = x.sum()
    sum_x2 = x2.sum()
    div = SLOPE_DENOTE_LEN * sum_x2 - sum_x ** 2

    for inst_type in INST_TYPE:
        try:
            dic[inst_type]['bidPrice1']
        except KeyError:
            #print(inst_type + " not exist")
            continue
        
        center = dic[inst_type][MEANPRICE][:, 0]
        Y = dic[inst_type][MEANPRICE][:, 0]#dic[inst_type]['bidPrice1']
        k = np.zeros_like(Y, dtype="float32")
        for i in range(Y.shape[0]-SLOPE_DENOTE_LEN):
            if Y[i] < 1:
                break
            y = Y[i:i+SLOPE_DENOTE_LEN]
            k[i] = (SLOPE_DENOTE_LEN * (x * y).sum() - sum_x * y.sum()) / div / 5000.0

        dic[inst_type][SLOPEPRICE] = k

def denote_dataset(dic):
    if CLASSIFICATION_METHOD == "slope":
        return denote_dataset_slope(dic)
    elif CLASSIFICATION_METHOD == "maxdiff":
        return denote_dataset_maxdiff(dic)

def denote_dataset_slope(dic):
    """
    Give label to sequence
    """
    for inst_type in INST_TYPE:
        try:
            dic[inst_type][SLOPEPRICE]
        except KeyError:
            #print(inst_type + " not exist")
            continue

        x = dic[inst_type][SLOPEPRICE]
        label = np.zeros_like(x, dtype="uint8")
        for i in range(x.shape[0]-TLEN):
            label[i] = get_class(x[i])
        dic[inst_type]['label'] = label

def denote_dataset_maxdiff(dic):
    """
    Give label to sequence by standard method
    """
    for inst_type in INST_TYPE:
        try:
            dic[inst_type]['bidPrice1']
        except KeyError:
            #print(inst_type + " not exist")
            continue

        x = dic[inst_type]['bidPrice1']
        label = np.zeros_like(x, dtype="uint8")
        for i in range(x.shape[0]-SLOPE_DENOTE_LEN):
            cur = x[i]
            maxi, mini = x[i:i+SLOPE_DENOTE_LEN].max(), x[i:i+SLOPE_DENOTE_LEN].min()
            pd = float(maxi - cur) / cur
            nd = float(cur - mini ) / cur
            label[i] = get_class_maxdiff(pd, nd)
            #print(pd, nd)
        dic[inst_type]['label'] = label

### Plot ###
############

def show_label(dic, contract_name="A1", st=0, ed=1000):
    colors = ['blue', 'cyan', 'grey', 'salmon', 'red']
    arr = dic[contract_name]["label"][st:ed]
    try:
        c = [colors[x-1] for x in arr]
    except TypeError:
        c = [colors[x[0]-1] for x in arr] 
    plt.scatter(range(st, ed), dic[contract_name]['bidPrice1'][st:ed], s=1, c=c)
    #plt.plot(range(st, ed), dic[contract_name][MEANPRICE][st:ed], c="r")
    plt.savefig("fig/" + contract_name + "label.png")
    plt.close()

def plot(y, name, color="r"):
    plt.plot(y, color)
    plt.savefig(name+".png")
    plt.close()

### IO ###
##########

def read_text(filename):
    strs = open(filename).read()

    secs = strs.split("Quote")[2:]
    raw_data = []
    for s in secs:
        if s[0] is not '[':
            continue

        ind = s.find("]")

        if ind == -1:
            continue
        
        raw_data.append(s[1:ind])
    return raw_data

def extract_data(raw_data, concise=False):
    INST_TYPE = ["A1", "A3", "B2", "B3"]
    dic = {}
    # init
    for t in INST_TYPE:
        dic[t] = {}
    
    inst_type = "A1"

    if concise:
        # 5 for bid; 7 for ask
        for line in raw_data:
            items = line.split(",")
            
            inst_type = items[-2][-2:]
            if inst_type not in INST_TYPE:
                continue

            k, v = items[5].split("=")
            v = int(v)
            if v == 0:
                continue

            try:
                dic[inst_type][k].append(v)
            except KeyError:
                dic[inst_type][k] = [v]
            k, v = items[7].split("=")
            try:
                dic[inst_type][k].append(v)
            except KeyError:
                dic[inst_type][k] = [v]

    else:
        for line in raw_data:
            items = line.split(",")
            inst_type = items[-2][-2:]
            if inst_type not in INST_TYPE:
                continue
            for t in items:
                left, right = t.split("=")
                if t[0] == "t":
                    # turnover is useless
                    continue
                if t[0] == "i":
                    # instrument ID
                    break
                try:
                    dic[inst_type][left].append(int(right))
                except KeyError:
                    dic[inst_type][left] = [int(right)]

    # make a numpy array
    for inst_type in INST_TYPE:
        for k in dic[inst_type].keys():
            dic[inst_type][k] = np.array(dic[inst_type][k])

    return dic

def get_dataset(filename, concise=False):
    """
    Dataset contract name: A1, A3, B2, B3
    For each contract name, there is all kinds of prices, each of them is a np.array sequence
    """
    raw_data = read_text(filename)
    dic = extract_data(raw_data, concise)
    get_mean_price(dic)
    get_fit_slope(dic)
    denote_dataset(dic)
    return dic

def compare_normalized(dic, contract_name="A1", comp=[""]):
    pass


### DEPRECATED ###
##################

def get_deal_price(dic):
    for inst_type in INST_TYPE:
        try:
            dic[inst_type]['lastPrice']
        except KeyError:
            #print(inst_type + " not exist")
            continue

        x = (dic[inst_type]['askPrice1'] + dic[inst_type]['bidPrice1']) / 2
        prev = x[0]
        for i in range(x.shape[0]):
            x[i] = (prev + x[i]) / 2
            prev = x[i]
        
        dic[inst_type]['dealPrice1'] = x