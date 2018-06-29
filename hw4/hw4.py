import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn.cluster
import tensorflow as tf

def default_data():
    A = [[1, 1, 1], [1, 1, 1]]
    B = [[3, 3, 3], [2, 3, 4]]
    C = [[7, 8, 9], [6, 6, 9]]
    
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)

    return A, B, C

def generate_data(pdfs, samples):
    """
    Mix pdfs generator with sample numbers
    """
    data = []
    for p, s in zip(pdfs, samples):
        class_data = []
        for i in range(p.shape[1]):
            class_data.append(np.random.normal(p[0][i], p[1][i], (s, 1)))
        class_data = np.concatenate(class_data, axis=1)
        data.append(class_data)
    data = np.concatenate(data, axis=0)
    return data

def scatter3d(data, name="dataset"):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=1)
    plt.savefig(name + ".png")
    plt.close()

def printable(data, N=3):
    for i in range(data.shape[0]):
        print("|%.5f|%.5f|%.5f|" % (data[i][0], data[i][1], data[i][2]))

def plot(data, name):
    plt.plot(data)
    plt.savefig(name+".png")
    plt.close()

def repeat_expr(func, params, N, draw=True, name="expr"):
    acc_collect = []
    loss_collect = []
    class_acc_collect = []
    for i in range(N):
        print("Repeat: %d" % i)
        data = generate_data(params, [1000, 600, 1600])
        label = np.array([0]*1000 + [1]*600 + [2]*1600)
        testdata = generate_data(params, [100, 100, 100])
        testlabel = np.array([0]*100 + [1]*100 + [2]*100)

        losses, acc, class_acc = func(data, label, testdata, testlabel)

        loss_collect.append(losses)
        class_acc_collect.append(class_acc)
        acc_collect.append(acc)
    
    if draw:
        loss_collect = np.array(loss_collect).mean(axis=0)
        plt.plot(loss_collect)
        plt.savefig(name + "_loss.png")
        plt.close()
    
    return get_statis_result(acc_collect, class_acc_collect)

def get_statis_result(acc_collect, class_acc_collect):
    cr = np.array(class_acc_collect) * 100
    r = np.array(acc_collect) * 100
    mean, std = r.mean(), r.std()
    print("Total Mean: %.2f\tStd: %.2f" % (mean, std))
    print("Class acc:")
    mean, std = cr.mean(axis=0), cr.std(axis=0)
    for i in range(3):
        print("%d Mean: %.2f\tStd: %.2f" % (i, mean[i], std[i]))
    return mean, std

def do_KNN(data, label, testdata, testlabel):
    knn = sklearn.neighbors.KNeighborsClassifier()
    knn.fit(data, label)
    #score = knn.score(testdata, testlabel)
    predict = knn.predict(testdata)
    count = (predict == testlabel).sum()
    acc = float(count) / float(testlabel.shape[0])

    class_acc = []
    for i in range(3):
        label_mask = (testlabel == i)
        count = ((predict == testlabel) * label_mask).sum()
        class_acc.append(float(count) / label_mask.sum())

    return knn, acc, class_acc

x = tf.placeholder(tf.float32, [None, 3])
y_true = tf.placeholder(tf.int32, [None,])

sess = tf.InteractiveSession()

def train(fetch, feed, N=1000):
    sess.run(tf.global_variables_initializer())
    losses = []
    for i in range(N):
        loss_, _ = sess.run(fetch, feed)
        losses.append(loss_)
    return np.array(losses)

def linear_classify(data, label, testdata, testlabel):
    # linear model
    w1 = tf.Variable(np.random.uniform(size=(3, 3)), dtype=tf.float32)
    b1 = tf.Variable(np.zeros((3,)), dtype=tf.float32)
    y_linear = tf.matmul(x, w1) + b1
    y_pred = tf.argmax(y_linear, axis=1)
    loss_linear = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_linear, labels=y_true))
    train_op_linear = tf.train.AdamOptimizer(0.01).minimize(loss_linear, var_list=[w1, b1])

    # random initialize 
    feed_train = {
        x: data.astype("float32"),
        y_true: label
    }
    feed_test = {
        x: testdata,
        y_true: testlabel
    }

    losses = train([loss_linear, train_op_linear], feed_train)

    predict = sess.run([y_pred], feed_test)[0]
    count = (predict == testlabel).sum()
    acc = float(count) / float(testlabel.shape[0])

    class_acc = []
    for i in range(3):
        label_mask = (testlabel == i)
        count = ((predict == testlabel) * label_mask).sum()
        class_acc.append(float(count) / label_mask.sum())

    return losses, acc, class_acc

def quad_classify(data, label, testdata, testlabel):
    # quad model
    w2 = tf.Variable(np.random.uniform(size=(3, 3)), dtype=tf.float32)
    w3 = tf.Variable(np.random.uniform(size=(3, 3, 3)), dtype=tf.float32)
    b2 = tf.Variable(np.zeros((3,)), dtype=tf.float32)

    y_quad = tf.reduce_sum(tf.tensordot(x, w3, [[1], [0]]) * tf.reshape(x, [-1, 3, 1]), axis=[1])
    y_quad += tf.matmul(x, w2) + b2
    #y_quad[:, i] += tf.reduce_sum(tf.matmul(x, w3[i]) * x, axis=1, keep_dims=True)
    y_pred = tf.argmax(y_quad, axis=1)
    loss_quad = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_quad, labels=y_true))
    train_op_quad = tf.train.AdamOptimizer().minimize(loss_quad, var_list=[w2, w3, b2])

    # random initialize 
    feed_train = {
        x: data.astype("float32"),
        y_true: label
    }
    feed_test = {
        x: testdata,
        y_true: testlabel
    }
    losses = train([loss_quad, train_op_quad], feed_train, 5000)

    predict = sess.run([y_pred], feed_test)[0]
    count = (predict == testlabel).sum()
    acc = float(count) / float(testlabel.shape[0])

    class_acc = []
    for i in range(3):
        label_mask = (testlabel == i)
        count = ((predict == testlabel) * label_mask).sum()
        class_acc.append(float(count) / label_mask.sum())

    return losses, acc, class_acc

# get dataset
params = default_data()

scatter3d(generate_data(params, [1000, 600, 1600]))

# KNN
print("KNN classification (5 repeats)")
result1 = repeat_expr(do_KNN, params, 5, False)

# linear discriminant
print("Linear classification (5 repeats)")
result2 = repeat_expr(linear_classify, params, 5, True, "linear")

# Quad discriminant
print("Quadratic classification (5 repeats)")
result3 = repeat_expr(quad_classify, params, 5, True, "Quad")