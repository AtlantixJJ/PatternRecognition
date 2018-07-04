import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.plot(range(10))
plt.close()
from os.path import join as pj
import scipy.interpolate as interp
import numpy as np
import lib

filename = pj("futuresData", "0-20170704-day.log")

def double_compare():
    dic1 = lib.get_dataset(pj("futuresData", "0-20170704-day.log"))
    dic2 = lib.get_dataset(pj("futuresData", "1-20170704-day.log"))

    l = []
    l.append(dic1['A1'][lib.MEANPRICE])
    l.append(dic1['A3'][lib.MEANPRICE])
    l.append(dic2['B2'][lib.MEANPRICE])
    l.append(dic2['B3'][lib.MEANPRICE])

    max_len = max([len(t) for t in l])
    for i in range(len(l)):
        if max_len == len(l[i]):
            l[i] = np.array(l[i], dtype="float64")
            continue

        y = np.array(l[i])[:, 0]
        x = np.arange(y.shape[0])

        f = interp.interp1d(x, y, 'linear')
        step = float(len(l[i])) / max_len    
        l[i] = np.array([f(step * t) for t in range(max_len - 1)], dtype="float64")
    
    for i in range(len(l)):
        m = l[i][1000:5000].max()
        print(m)
        plt.plot(l[i][1000:5000] / m)
    plt.legend(["A1", "A3", "B2", "B3"])
    plt.savefig("a.png")
    plt.close()

    return l

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
    lib.plot(diff_ask_bid, pj("fig", "A1_diff_askPrice1_bidPrice1"))

def plot_contract(dic, contract_name="A1", st=0, ed=1000):
    for k, v in dic[contract_name].items():
        plt.plot(v[st:ed])
        plt.savefig("fig/" + contract_name + "_" + k + "_" + str(st) + "_" + str(ed) + ".png")
        plt.close()

def plot_compare(dic, contract_name=["A1", "A3", "B2", "B3"], st=0, ed=1000, comp=["askPrice1", "bidPrice1"], ifNorm=True):
    colors = "rgbc"
    for cname in contract_name:
        name = cname
        if len(dic[cname].keys()) < 2:
            print(cname + " not exist")
            continue

        fig, ax1 = plt.subplots()

        for i, k in enumerate(comp):
            c = colors[i]
            name = name + "_" + k
            if ifNorm and k.find("slope") == -1:
                x = dic[cname][k][st:ed].astype("float32")
                ax1.plot(range(st, ed), x, c)
                #plt.plot(x, c)
            else:
                ax2 = ax1.twinx()
                #ax2.set_ylim([-0.003, 0.003])
                ax2.plot(range(st, ed), dic[cname][k][st:ed], c+".", markersize=1)
        
        plt.savefig("fig/" + name + "_" + str(st) + "_" + str(ed) + ".png")
        plt.close()

def print_contract(dic, contract_name='A1', st=0, ed=1000, printItems=["askPrice1", "bidPrice1"]):
    pass


def get_figures(dic):
    lib.analysis_diff_ask_bid(dic)
    plot_contract(dic, st=5000, ed=6000)

#l = double_compare()

raw_data = lib.read_text(filename)

dic = lib.extract_data(raw_data)

lib.get_mean_price(dic)
lib.get_fit_slope(dic)
lib.denote_dataset(dic)
lib.show_label(dic, contract_name="A1", st=5000, ed=7000)
lib.show_label(dic, contract_name="A3", st=5000, ed=10000)
print(dic["A1"].keys())
plot_compare(dic, comp=['bidPrice1', lib.MEANPRICE, lib.SLOPEPRICE], st=5000, ed=10000, ifNorm=True)
plot_compare(dic, comp=['askPrice1', 'bidPrice1'], st=0, ed=1000, ifNorm=False)
analysis_diff_ask_bid(dic)
#get_figures(dic)