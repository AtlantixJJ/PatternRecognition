import sys
sys.path.insert(0, "src")
import dataset
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorpack as tp

INPUT_LEN = dataset.INPUT_LEN
OUTPUT_LEN = dataset.OUTPUT_LEN
BATCH_SIZE = 64
N_EPOCH = 20
EPSILON = 1e-6

#DATA_DIR = "futuresData"
#futures_data = dataset.FuturesData(DATA_DIR)
#futures_data.save_npz()

def mlp_2500(x):
    # x: (-1, 4, 200)
    # mean: (-1, 4, 1)
    x /= 5000.0
    mean, var = tf.nn.moments(x, 2, keep_dims=True)
    std = tf.sqrt(var) + EPSILON
    x = (x - mean) / std

    gate = tf.Variable(np.ones((1, 4, 1), "float32"))
    x *= tf.nn.sigmoid(gate)

    x = tf.reshape(x, [-1, 4 * INPUT_LEN])

    x = tf.layers.dense(x, 1024, tf.nn.relu)
    x = tf.layers.dense(x, 4 * OUTPUT_LEN, None)
    #x = tf.layers.dense(x, OUTPUT_LEN, None)    
    
    x = tf.reshape(x, [-1, 4, OUTPUT_LEN])
    x = x * std + mean
    x *= 5000.0
    return x

def mlp(x):
    # x: (-1, 4, 200)
    # mean: (-1, 4, 1)
    x /= 5000.0
    mean, var = tf.nn.moments(x, 2, keep_dims=True)
    std = tf.sqrt(var) + EPSILON
    x = (x - mean) / std
    x = tf.reshape(x, [-1, 4 * INPUT_LEN])

    out_tensors = []
    for i in range(4):
        tmp = tf.layers.dense(x, 1024, tf.nn.relu)
        tmp = tf.layers.dense(tmp, OUTPUT_LEN, None)
        tmp = tmp * std[:, i, :] + mean[:, i, :]
        out_tensors.append(tmp)
    x = tf.stack(out_tensors, 1)

    x *= 5000.0
    return x

def linear(x):
    # x: (-1, 4, 200)
    x /= 5000.0
    x = tf.reshape(x, [-1, INPUT_LEN])
    mean, var = tf.nn.moments(x, 1, keep_dims=True)
    std = tf.sqrt(var) + EPSILON
    x = (x - mean) / std
    #x = tf.layers.dense(x, 512, tf.nn.relu)
    x = tf.layers.dense(x, OUTPUT_LEN, None)
    x = x * std + mean
    x *= 5000
    #x = tf.reshape(x, [-1, 4, 40])
    return x

def train_epoch(fetches, feeds, dl, sess, learning_rate=0.001):
    dl.reset_state()

    x, y, lr = feeds
    est_y, loss, optim = fetches

    for idx, sample in enumerate(dl.generator()):
        sample = sample[0]
        train_seq = np.array([np.stack(sample[i, 0, :]) for i in range(sample.shape[0])])
        label_seq = np.array([np.stack(sample[i, 1, :]) for i in range(sample.shape[0])])

        #loss_, _ = sess.run([loss, optim], {x: train_seq, y: label_seq[:, 0, :]})
        loss_, _ = sess.run([loss, optim], {x: train_seq, y: label_seq, lr: learning_rate})
        print(loss_)

def test_mlp():
    # build dataset
    dl = dataset.DataLoader([dataset.FuturesData(from_npz=True)], BATCH_SIZE)

    # control var
    lr = tf.placeholder(tf.float32, [], "lr")

    # build network
    x = tf.placeholder(tf.float32, [None, 4, INPUT_LEN], name="x")
    y = tf.placeholder(tf.float32, [None, 4, OUTPUT_LEN], name="x")
    est_y = mlp(x)

    regression_loss_inst = tf.reduce_mean(tf.abs(est_y - y), axis=[0, 2])
    regression_loss = tf.reduce_mean(regression_loss_inst)
    optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(regression_loss)

    # init
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_id in range(N_EPOCH):
            if epoch_id < 10:
                LR = 0.001
            else:
                LR = 0.0001
            train_epoch([est_y, regression_loss_inst, optim], [x, y, lr], dl, sess, LR)

    dl.reset_state()

def test_linear():
    tp.logger.info("Test linear")

    # build dataset
    dl = dataset.DataLoader([dataset.FuturesData(from_npz=True)], BATCH_SIZE)

    # control var
    lr = tf.placeholder(tf.float32, [], "lr")

    # build network
    x = tf.placeholder(tf.float32, [None, INPUT_LEN], name="x"); y = tf.placeholder(tf.float32, [None, OUTPUT_LEN], name="x")
    est_y = linear(x)
    regression_loss_inst = tf.reduce_mean(tf.abs(est_y - y), axis=[0, 1])
    regression_loss = tf.reduce_mean(regression_loss_inst)
    optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(regression_loss)

    # init
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_id in range(N_EPOCH):
            if epoch_id < 10:
                LR = 0.001
            else:
                LR = 0.0001

            dl.reset_state()
            for idx, sample in enumerate(dl.generator()):
                sample = sample[0]
                train_seq = np.array([np.stack(sample[i, 0, :]) for i in range(sample.shape[0])])
                label_seq = np.array([np.stack(sample[i, 1, :]) for i in range(sample.shape[0])])

                #loss_, _ = sess.run([loss, optim], {x: train_seq, y: label_seq[:, 0, :]})
                loss_, _ = sess.run([regression_loss_inst, optim],
                                    {x: train_seq[:, 3, :], y: label_seq[:, 3, :], lr: LR})
                print(loss_)

    dl.reset_state()



if __name__ == "__main__":
    test_linear()