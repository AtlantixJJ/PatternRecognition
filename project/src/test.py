import sys
sys.path.insert(0, "src")
import dataset
import numpy as np
import pandas as pd
import tensorflow as tf

INPUT_LEN = dataset.INPUT_LEN
OUTPUT_LEN = dataset.OUTPUT_LEN
BATCH_SIZE = 64
N_EPOCH = 20

#DATA_DIR = "futuresData"
#futures_data = dataset.FuturesData(DATA_DIR)
#futures_data.save_npz()

def mlp(x):
    # x: (-1, 4, 200)
    x = tf.reshape(x, [-1, 4 * INPUT_LEN])
    x = tf.layers.dense(x, 512, tf.nn.relu)
    x = tf.layers.dense(x, 40, None)
    return x

def train_epoch(fetches, feeds, dl, sess):
    dl.reset_state()

    x, y = feeds
    est_y, loss, optim = fetches

    for idx, sample in enumerate(dl.generator()):
        sample = sample[0]
        train_seq = np.array([np.stack(sample[i, 0, :]) for i in range(sample.shape[0])])
        label_seq = np.array([np.stack(sample[i, 1, :]) for i in range(sample.shape[0])])
        
        loss_, _ = sess.run(fetches, {x: train_seq, y: label_seq})
        print(loss_)

if __name__ == "__main__":
    # build dataset
    dl = dataset.DataLoader([dataset.FuturesData(from_npz=True)], BATCH_SIZE)

    # build network
    x = tf.placeholder(tf.float32, [None, 4, INPUT_LEN], name="x")
    y = tf.placeholder(tf.float32, [None, 4, OUTPUT_LEN], name="x")
    est_y = mlp(x)
    regression_loss = tf.reduce_mean(tf.abs(est_y - y))
    optim = tf.train.AdamOptimizer().minimize(regression_loss)

    # init
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_id in range(N_EPOCH):
            train_epoch([est_y, regression_loss, optim], [x, y], dl, sess)

    dl.reset_state()