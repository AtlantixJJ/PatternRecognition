import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.plot(range(10))
plt.close()

import numpy as np
import sklearn.linear_model
import lib, train
import os

DATA_DIR = "futuresData/"

def testall(method):
    all_acc = []
    all_err = []
    result_dic = {}
    err_dic = {}
    dirs = os.listdir(DATA_DIR)
    dirs.sort()
    for f in dirs:
        filename = DATA_DIR + f
        dic = lib.get_dataset(filename, True)
        for k in dic.keys():
            if len(dic[k].keys()) > 1 and dic[k][lib.MEANPRICE].shape[0] > 1000:

                acc, err = test_alg(dic[k], method)
                name = f[:-4] + "_" + k

                result_dic[name] = acc
                if err != -1:
                    err_dic[name] = err

                all_acc.append(acc)
                all_err.append(err)
                print("%s %.5f %.5f" % (name, acc, err))

    np.savez(method, [result_dic, err_dic])
    return result_dic, err_dic

def summary_dic(resdic):
    all_vals = np.array(list(resdic.values()))

    # summary up INST TYPE
    INST_TYPE = ["A1", "A3", "B2", "B3"]
    inst_vals = []
    for t in INST_TYPE:
        acc = []
        for k, v in resdic.items():
            if t in k:
                acc.append(v)
        acc = np.array(acc).mean()
        inst_vals.append(acc)
    
    # summary up day and night
    DAY_TYPE = ["day", "night"]
    day_vals = []
    for t in DAY_TYPE:
        acc = []
        for k, v in resdic.items():
            if t in k:
                acc.append(v)
        acc = np.array(acc).mean()
        day_vals.append(acc)

    return all_vals.mean(), inst_vals, day_vals

def test_alg(dic, method):
    if method == "SMOOTHLINEAR":
        reg, train_X, test_X, train_Y, test_Y, test_label = train.linear_fit(dic)
        acc, err = train.testacc_linear(reg, test_X, test_Y, test_label)

    elif method == "LINEAR":
        reg, train_X, test_X, train_Y, test_Y, test_label = train.linear_fit(dic)
        acc, err = train.testacc_linear(reg, test_X, test_Y, test_label, smooth=False)

    elif method == "LOGISTIC":
        reg, train_X, test_X, train_Y, test_Y = train.linear_classify(dic)
        acc = train.testacc_logistic(reg, test_X, test_Y)
        err = -1
    
    return acc, err

def analyze_from_npz(fname):
    result_dic, err_dic = np.load(fname, encoding='latin1')['arr_0'].tolist()
    full_avg, inst_avg, day_avg = summary_dic(result_dic)
    full_err_avg, inst_err_avg, day_err_avg = summary_dic(err_dic)
    return [result_dic, err_dic, full_avg, inst_avg, day_avg, full_err_avg, inst_err_avg, day_err_avg]

if __name__ == "__main__":
    for method in ["SMOOTHLINEAR", "LINEAR", "LOGISTIC"]:
        result_dic, err_dic = testall(method)
        full_avg, inst_avg, day_avg = summary_dic(result_dic)
        full_err_avg, inst_err_avg, day_err_avg = summary_dic(err_dic)
        print("method %s: %f" % (method, full_avg))
