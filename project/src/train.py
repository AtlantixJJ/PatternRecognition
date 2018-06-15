import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.plot(range(10))
plt.close()
import sklearn.linear_model
import numpy as np
import lib

MEANPRICE = lib.MEANPRICE
TLEN = lib.TLEN
INPUT_LEN = 100 # the length of network input
PREDICT_LEN = 40
TRAIN_PORTION = 0.8
DEP_LEN = TLEN+INPUT_LEN+PREDICT_LEN

CLASSIFICATION_METHOD = lib.CLASSIFICATION_METHOD

def lienar2class(Y):
    if CLASSIFICATION_METHOD == "slope":
        return linear2class_slope(Y)
    elif CLASSIFICATION_METHOD == "maxdiff":
        return linear2class_maxdiff(Y)

def linear2class_slope(Y):
    """
    Y's shape should be (N_SAMPLE, PREDICT_LEN)
    """
    x = np.array(range(0, PREDICT_LEN))
    x2 = x ** 2
    sum_x = x.sum()
    sum_x2 = x2.sum()
    div = PREDICT_LEN * sum_x2 - sum_x ** 2

    est_label = np.zeros((Y.shape[0]-PREDICT_LEN))
    for i in range(Y.shape[0]-PREDICT_LEN):
        if Y[i] < 1:
            break
        y = Y[i:i+PREDICT_LEN]
        k = ((PREDICT_LEN * (x * y).sum() - sum_x * y.sum()) / div) * 10000.0 / float(y[0])
        est_label[i] = lib.get_class(k)
    
    return est_label

def linear2class_maxdiff(Y):
    """
    Y's shape should be (N_SAMPLE, PREDICT_LEN)
    """
    est_label = np.zeros_like(Y, dtype="uint8")
    for i in range(Y.shape[0]-TLEN):
        cur = Y[i]
        maxi, mini = Y[i:i+TLEN].max(), Y[i:i+TLEN].min()
        pd = float(maxi - cur) / cur
        nd = float(cur - mini ) / cur
        est_label[i] = lib.get_class_maxdiff(pd, nd)
    
    return est_label

def accuracy(est_label, label):
    count = (est_label == label).sum()
    plt.plot(est_label[:1000], 'r')
    plt.plot(label[:1000], 'b')
    plt.savefig("test.png")
    plt.close()
    return float(count) / label.shape[0]

def smooth_predict(reg, dataset_X):
    est_Y = reg.predict(dataset_X)
    avg_Y = np.zeros((est_Y.shape[0]), dtype="float32")
    for i in range(est_Y.shape[0]):
        LEN = min(est_Y.shape[0] - i, PREDICT_LEN)
        avg_Y[i:i+PREDICT_LEN] += est_Y[i][:LEN]
    for i in range(PREDICT_LEN):
        avg_Y[i] /= float(i+1)
    avg_Y[PREDICT_LEN:-PREDICT_LEN] /= float(PREDICT_LEN)
    return avg_Y

def testacc_linear(reg, dataset_X, dataset_Y, label, smooth=True):
    print("Test accuracy of linear regression: ")

    if smooth:
        est_Y = smooth_predict(reg, dataset_X)
    else:
        est_Y = reg.predict(dataset_X[::PREDICT_LEN]).reshape(-1)

    est_label = lienar2class(est_Y)

    acc = accuracy(est_label[:label.shape[0]], label)
    err_list = np.abs(dataset_Y - est_Y[:dataset_Y.shape[0]])
    err = err_list.mean()
    err_std = (err_list - err).std()
    print("Err %f Std %f" % (err, err_std))
    return acc, err

def testacc_logistic(reg, dataset_X, label):
    print("Test accuracy of logistic regression: ")

    est_label = reg.predict(dataset_X)
    acc = accuracy(est_label[:label.shape[0]], label)
    return acc

def show_result_linear(reg, dataset_X, dataset_Y, st=0, ed=16000, name="linearfit", smooth=True):
    if smooth:
        trace_Y = smooth_predict(reg, dataset_X)
    else:
        trace_Y = reg.predict(dataset_X[::PREDICT_LEN]).reshape(-1)

    MINLEN = min(trace_Y.shape[0], dataset_Y.shape[0])

    plt.plot(range(st, ed), dataset_Y[st:ed], 'r')
    plt.plot(range(st, ed), trace_Y[st:ed], 'b')
    plt.savefig("fig/" + name + str(st) + "_" + str(ed) + ".png")
    plt.close()

    loss = (dataset_Y[:MINLEN] - trace_Y[:MINLEN]).std()
    print("Point loss: %.5f" % loss)

def get_linear_dataset(dic):
    raw_seq = dic[MEANPRICE]
    X = []
    # 0~INPUT_LEN -> INPUT_LEN+1~INPUT_LEN+PREDICT_LEN
    for i in range(raw_seq.shape[0]-DEP_LEN):
        X.append(raw_seq[i:i+INPUT_LEN])
    X = np.array(X)

    border = int(X.shape[0] * TRAIN_PORTION)

    Y = []
    for i in range(raw_seq.shape[0]-DEP_LEN):
        Y.append(raw_seq[INPUT_LEN+i: INPUT_LEN+i+PREDICT_LEN])
    Y = np.array(Y)

    train_X, test_X = X[:border], X[border:]
    train_Y, label = Y[:border], dic['label'][INPUT_LEN + border: INPUT_LEN + border + test_X.shape[0] - PREDICT_LEN]
    test_Y = raw_seq[INPUT_LEN+border:INPUT_LEN + border + test_X.shape[0] - PREDICT_LEN]
    return train_X, test_X, train_Y, test_Y, label

def get_logistic_dataset(dic):
    raw_seq = dic[MEANPRICE]
    X = []
    for i in range(raw_seq.shape[0]-DEP_LEN):
        X.append(raw_seq[i:i+INPUT_LEN])
    X = np.array(X)

    border = int(X.shape[0] * TRAIN_PORTION)

    train_X, test_X = X[:border], X[border:]
    train_Y, test_Y = dic['label'][INPUT_LEN:INPUT_LEN + border], dic['label'][INPUT_LEN + border:INPUT_LEN + border + test_X.shape[0] - PREDICT_LEN]
    return train_X, test_X, train_Y, test_Y

def linear_fit(dic):
    model = sklearn.linear_model.LinearRegression()

    train_X, test_X, train_Y, test_Y, label = get_linear_dataset(dic)

    print(train_X.shape, train_Y.shape)
    print(test_X.shape, test_Y.shape)

    reg = model.fit(train_X, train_Y)

    return reg, train_X, test_X, train_Y, test_Y, label

def linear_classify(dic):
    model = sklearn.linear_model.LogisticRegression()

    train_X, test_X, train_Y, test_Y = get_logistic_dataset(dic)

    print(train_X.shape, train_Y.shape)
    print(test_X.shape, test_Y.shape)

    reg = model.fit(train_X, train_Y)

    return reg, train_X, test_X, train_Y, test_Y