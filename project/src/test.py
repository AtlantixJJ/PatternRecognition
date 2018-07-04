import sys
sys.path.insert(0, "src")
import dataset, lib
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorpack as tp
from os.path import join as pj

INST_TYPE = lib.INST_TYPE
INPUT_LEN = dataset.INPUT_LEN
OUTPUT_LEN = dataset.OUTPUT_LEN
BATCH_SIZE = 64
N_EPOCH = 80
STAIRCASE = 15
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

def mlp_combine(x):
    # x: (-1, 4, 200)
    # mean: (-1, 4, 1)
    x /= 5000.0
    mean, var = tf.nn.moments(x, 2, keep_dims=True)
    std = tf.sqrt(var) + EPSILON
    x = (x - mean) / std
    x = tf.reshape(x, [-1, 4 * INPUT_LEN])

    out_tensors = []
    for i in range(4):
        tmp = tf.layers.dense(x, 1024, tf.nn.tanh)
        tmp = tf.layers.dense(tmp, OUTPUT_LEN, None)
        tmp = tmp * std[:, i, :] + mean[:, i, :]
        out_tensors.append(tmp)
    x = tf.stack(out_tensors, 1)

    x *= 5000.0
    return x

def mlp_single(x):
    x /= 5000.0
    x = tf.reshape(x, [-1, INPUT_LEN])
    mean, var = tf.nn.moments(x, 1, keep_dims=True)
    std = tf.sqrt(var) + EPSILON
    x = (x - mean) / std
    x = tf.layers.dense(x, 1024, tf.nn.tanh)
    x = tf.layers.dense(x, OUTPUT_LEN, None)
    x = x * std + mean
    x *= 5000
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

def test_combine(name="combine"):
    # build dataset
    train_dl = dataset.DataLoader([dataset.FuturesData(is_train=True, from_npz=True)], BATCH_SIZE)
    val_dl = dataset.DataLoader([dataset.FuturesData(is_train=False, from_npz=True)], BATCH_SIZE)

    # control var
    lr = tf.placeholder(tf.float32, [], "lr")

    # build network
    x = tf.placeholder(tf.float32, [None, 4, INPUT_LEN], name="x")
    y = tf.placeholder(tf.float32, [None, 4, OUTPUT_LEN], name="x")
    est_y = mlp_combine(x)

    regression_loss_inst = tf.reduce_mean(tf.abs(est_y - y), axis=[0, 2])
    regression_loss = tf.reduce_mean(regression_loss_inst)
    optim = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(regression_loss)

    inst_summary = [tf.summary.scalar("regression/inst%s" % INST_TYPE[i], regression_loss_inst[i]) for i in range(4)]
    summary = tf.summary.merge(inst_summary)
    summary_writer = tf.summary.FileWriter("logs/" + name)

    saver = tf.train.Saver()

    # init
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        global_iter = 0
        for epoch_id in range(N_EPOCH):
            LR = 0.1 ** (N_EPOCH // STAIRCASE)

            train_dl.reset_state()
            for idx, sample in enumerate(train_dl.generator()):
                sample = sample[0]
                train_seq = np.array([np.stack(sample[i, 0, :]) for i in range(sample.shape[0])])
                label_seq = np.array([np.stack(sample[i, 1, :]) for i in range(sample.shape[0])])

                #loss_, _ = sess.run([loss, optim]LR, {x: train_seq, y: label_seq[:, 0, :]})
                sum_, _ = sess.run([summary, optim], {x: train_seq, y: label_seq, lr: LR})
                summary_writer.add_summary(sum_, global_iter)
                global_iter += 1
        
            save_path = saver.save(sess, pj("model", name + ".ckpt"))
            print("Model saved in %s" % save_path)

def test_single(is_linear=True, name="single"):
    tp.logger.info("Test linear")

    # build dataset
    train_dl = dataset.DataLoader([dataset.FuturesData(is_train=True, from_npz=True)], BATCH_SIZE)
    val_dl = dataset.DataLoader([dataset.FuturesData(is_train=False, from_npz=True)], BATCH_SIZE)

    # control var
    lr = tf.placeholder(tf.float32, [], "lr")

    # build network
    x = tf.placeholder(tf.float32, [None, INPUT_LEN], name="x")
    y = tf.placeholder(tf.float32, [None, OUTPUT_LEN], name="x")

    summary_writer = tf.summary.FileWriter("logs/" + name)

    # init
    with tf.Session() as sess:
        for inst_type in range(4):
            global_iter = 0

            if is_linear:
                est_y = linear(x)
            else:
                est_y = mlp_single(x)
            saver = tf.train.Saver()
            regression_loss = tf.reduce_mean(tf.abs(est_y - y))
            optim = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(regression_loss)
            sess.run(tf.global_variables_initializer())

            inst_summary = tf.summary.scalar("regression/inst%s" % INST_TYPE[inst_type], regression_loss)

            for epoch_id in range(N_EPOCH):
                LR = 0.1 ** (N_EPOCH // STAIRCASE)

                train_dl.reset_state()
                for idx, sample in enumerate(train_dl.generator()):
                    sample = sample[0]
                    train_seq = np.array([np.stack(sample[i, 0, :]) for i in range(sample.shape[0])])
                    label_seq = np.array([np.stack(sample[i, 1, :]) for i in range(sample.shape[0])])

                    #loss_, _ = sess.run([loss, optim], {x: train_seq, y: label_seq[:, 0, :]})
                    sum_, _ = sess.run([inst_summary, optim],
                                        {
                                            x: train_seq[:, inst_type, :],
                                            y: label_seq[:, inst_type, :],
                                            lr: LR
                                        })
                    summary_writer.add_summary(sum_, global_iter)
                    global_iter += 1
            
                save_path = saver.save(sess, pj("model", name + "_" + str(inst_type) + ".ckpt"))
                print("Model saved in %s" % save_path)

if __name__ == "__main__":
    #saver.restore(sess, "/tmp/model.ckpt")
    if sys.argv[1] == "1":
        test_single(True, "single_linear")
    elif sys.argv[1] == "2":
        test_single(False, "single_mlp")
    elif sys.argv[1] == "3":
        test_combine("combine_mlp")