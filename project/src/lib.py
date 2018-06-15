import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.plot(range(10))
plt.close()

import numpy as np

INST_TYPE = ["A1", "A3", "B2", "B3"]
THRESHOLD1 = 0.05 #0.002
THRESHOLD2 = 0.10 #0.004
TLEN = 200

MEANPRICE = "meanPrice200"

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

def get_mean_price(dic):
    """
    Smooth the data by averaging with window size 200.
    """
    WIN_SIZE = 200
    for inst_type in INST_TYPE:
        try:
            dic[inst_type]['bidPrice1']
        except KeyError:
            #print(inst_type + " not exist")
            continue
        
        x = dic[inst_type]['bidPrice1']
        m = np.zeros_like(x)
        for i in range(x.shape[0]-WIN_SIZE):
            m[i] = x[i:i+WIN_SIZE].mean()
        dic[inst_type]['meanPrice'+str(WIN_SIZE)] = m

def get_fit_slope(dic):
    """
    Compute slope in future 40 points in order to determine its tendency.
    """
    WIN_SIZE = 40
    x = np.array(range(0, WIN_SIZE))
    x2 = x ** 2
    sum_x = x.sum()
    sum_x2 = x2.sum()
    div = WIN_SIZE * sum_x2 - sum_x ** 2

    for inst_type in INST_TYPE:
        try:
            dic[inst_type]['bidPrice1']
        except KeyError:
            #print(inst_type + " not exist")
            continue
        
        Y = dic[inst_type]['meanPrice200']
        k = np.zeros_like(Y, dtype="float32")
        for i in range(Y.shape[0]-WIN_SIZE):
            if Y[i] < 1:
                break
            y = Y[i:i+WIN_SIZE]
            k[i] = ((WIN_SIZE * (x * y).sum() - sum_x * y.sum()) / div) * 10000.0 / Y[i]

        dic[inst_type]['slopePrice'+str(WIN_SIZE)] = k

def denote_dataset(dic):
    """
    Give label to sequence
    """
    for inst_type in INST_TYPE:
        try:
            dic[inst_type]['slopePrice40']
        except KeyError:
            #print(inst_type + " not exist")
            continue

        x = dic[inst_type]['slopePrice40']
        label = np.zeros_like(x, dtype="uint8")
        for i in range(x.shape[0]-200):
            label[i] = get_class(x[i])
        dic[inst_type]['label'] = label

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
            try:
                dic[inst_type][k].append(int(v))
            except KeyError:
                dic[inst_type][k] = [int(v)]
            k, v = items[7].split("=")
            try:
                dic[inst_type][k].append(int(v))
            except KeyError:
                dic[inst_type][k] = [int(v)]

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

def show_label(dic, contract_name="A1", st=0, ed=1000):
    colors = ['blue', 'cyan', 'grey', 'salmon', 'red']
    arr = dic[contract_name]["label"][st:ed]
    c = [colors[x-1] for x in arr]
    plt.scatter(range(st, ed), dic[contract_name]['bidPrice1'][st:ed], s=1, c=c)
    plt.plot(range(st, ed), dic[contract_name]['meanPrice200'][st:ed], c="r")
    plt.savefig("fig/" + contract_name + "label.png")
    plt.close()

def compare_normalized(dic, contract_name="A1", comp=[""]):
    pass

def plot(y, name, color="r"):
    plt.plot(y, color)
    plt.savefig(name+".png")
    plt.close()

def analysis_diff_ask_bid(dic):
    diff_ask_bid = dic['A1']['askPrice1'] - dic['A1']['bidPrice1'] 
    count = 0
    for i in range(diff_ask_bid.shape[0]):
        if diff_ask_bid[i] > 5000:
            count += 1
            diff_ask = dic['A1']['askPrice1'][i] - dic['A1']['askPrice1'][i-1]
            diff_bid = dic['A1']['bidPrice1'][i] - dic['A1']['bidPrice1'][i-1]
            print("%d %d = %d %d" % (diff_ask_bid[i-1], diff_ask_bid[i], diff_ask, diff_bid))

    print("%d %f" % (count, float(count) / diff_ask_bid.shape[0]))
    plot(diff_ask_bid, "fig/A1_diff_askPrice1_bidPrice1")

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