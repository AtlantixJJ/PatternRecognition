#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import os
import random

from datetime import datetime
from pathlib import Path
from sklearn import preprocessing
from sklearn.externals import joblib

# import util
from util import IDs, keywords

data_path = '../data/'
scalerX_path = '../data/scalerX.pkl'
scalerY_path = '../data/scalerY.pkl'

test_start_date = datetime.strptime('20170810', "%Y%m%d")

time_step = look_back = 36
look_up = 4
rnn_unit = 1000
batch_size = 3000
valid_size = 200
input_size = len(IDs) * (len(keywords)-1)
output_size = len(IDs)
# TODO dynamic learning rate
lr = 0.0006

k = 0.3


now = datetime.now()
logdir = './logs_' + now.strftime("%m%d-%H%M%S") + '/'
modeldir = './model_' + now.strftime("%m%d-%H%M%S") + '/'
load_modeldir = './model_0703-001011/'


def readData(data_path, look_back, look_up):
    mat_files = [ f for f in os.listdir(data_path)
                 if os.path.isfile(data_path+f) and Path(f).suffix == '.txt']

    dataX = []
    dataY = []
    testX = []
    testY = []
    for f in mat_files:
        data = np.loadtxt(data_path + f)
        # reshape matrix from (times*IDs, keywords) to (times, IDs, keywords)
        data = np.reshape(data, (-1, len(IDs), len(keywords)))
        # get volumn data
        volumn = data[:, :, 3]
        volumn = np.diff(volumn, axis=0)
        data = data[1:, :, :]
        data[:, :, 3] = volumn
        for i in range(len(data) - (look_back + look_up)):
            # remove P05s, this should be result value
            x = np.delete(data[i:i + look_back, :, :], obj=4, axis=2)
            y = data[i+look_up: i+look_back+look_up, :, 4]
            if ( datetime.strptime(f[0:8], "%Y%m%d") < test_start_date ):
                dataX.append(x)
                dataY.append(y)
            else:
                testX.append(x)
                testY.append(y)

    standard_scalerX = preprocessing.StandardScaler()
    standard_scalerY = preprocessing.StandardScaler()
    # reshape it to 2 dim
    dataX = np.reshape(dataX, (-1, len(IDs)*(len(keywords)-1)))
    testX = np.reshape(testX, (-1, len(IDs)*(len(keywords)-1)))
    dataY = np.reshape(dataY, (-1, len(IDs)))
    testY = np.reshape(testY, (-1, len(IDs)))
    dataX = standard_scalerX.fit_transform(dataX)
    dataY = standard_scalerY.fit_transform(dataY)
    testX = standard_scalerX.transform(testX)
    testY = standard_scalerY.transform(testY)
    dataX = np.reshape(dataX, (-1, time_step, len(IDs), len(keywords)-1))
    dataY = np.reshape(dataY, (-1, time_step, len(IDs)))
    testX = np.reshape(testX, (-1, time_step, len(IDs), len(keywords)-1))
    testY = np.reshape(testY, (-1, time_step, len(IDs)))
    joblib.dump(standard_scalerX, scalerX_path)
    joblib.dump(standard_scalerY, scalerY_path)
    return (dataX, dataY, testX, testY, standard_scalerX, standard_scalerY)

def setup_lstm():
    X = tf.placeholder(tf.float32, [None, time_step, input_size], name="X")
    Y = tf.placeholder(tf.float32, [None, time_step, output_size], name="Y")
    return (X, Y)

def lstm(batch, X, Y):
    # TODO initialize variables with none zero data
    with tf.name_scope('Wx_plus_b_input'):
        with tf.variable_scope("W_b_in", reuse=tf.AUTO_REUSE):
            w_in = tf.get_variable('W_in', [input_size, rnn_unit])
            b_in = tf.get_variable('b_in', [rnn_unit,])
    input_x = tf.reshape(X, [-1, input_size])
    input_rnn = tf.matmul(input_x, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])

    with tf.name_scope('LSTM_cell'):
        with tf.variable_scope("BasicLSTMcell", reuse=tf.AUTO_REUSE):
            cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
            init_state = cell.zero_state(batch, dtype=tf.float32)
            output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn,
                                initial_state=init_state, dtype=tf.float32)
    with tf.name_scope('Wx_plus_b_output'):
        output = tf.reshape(output_rnn, [-1, rnn_unit])
        with tf.variable_scope('W_b_out', reuse=tf.AUTO_REUSE):
            w_out = tf.get_variable('W_out', [rnn_unit, output_size])
            b_out = tf.get_variable('b_out', [output_size,])
        pred = tf.matmul(output, w_out) + b_out
        pred = tf.reshape(pred, [batch, time_step, -1], name="prediction")

    with tf.name_scope('loss'):
        # pred = pred[:, -look_up:, :]
        sum_loss = tf.reduce_sum(tf.square(tf.reshape(pred, [batch, -1]) -
                                           tf.reshape(Y, [batch, -1])), 1)
        loss = tf.reduce_mean(sum_loss)

    return (pred, loss, final_states)


def train_lstm(trainX, trainY, validX, validY, recover=False):
    with tf.Graph().as_default():
        with tf.name_scope('Train'):
            X, Y = setup_lstm()
            pred, loss, _ = lstm(batch_size, X, Y)
            train_op = tf.train.AdamOptimizer(lr).minimize(loss)
            tf.summary.scalar('Training Loss', loss)
        with tf.name_scope('Valid'):
            valid_X, valid_Y = setup_lstm()
            valid_pred, valid_loss, _ = \
                lstm(valid_size, valid_X, valid_Y)
            tf.summary.scalar("Validation Loss", valid_loss)

        merged = tf.summary.merge_all()

        saver = tf.train.Saver(tf.global_variables())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            if (recover == True):
                module_file = tf.train.latest_checkpoint(load_modeldir)
                saver.restore(sess, module_file)
            else:
                sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(logdir, sess.graph)
            for i in range(6000):
                start = 0
                end = start + batch_size
                while (end < len(trainX)):
                    _, loss_ = sess.run([train_op, loss], feed_dict={
                        X:np.reshape(trainX[start:end], (-1, time_step, input_size)),
                        Y:np.reshape(trainY[start:end], (-1, time_step, output_size)) })
                    start += batch_size
                    end = start + batch_size
                print("%dth epoch" % i)
                if i % 10 == 0:
                    print("epoch: " + str(i) + ", loss: " + str(loss_))
                    print("save model: " + saver.save(sess, modeldir + "predict.model"))
                    valid_loss_, merged_ = sess.run([valid_loss, merged], feed_dict={
                        valid_X:np.reshape(validX[0:valid_size], (-1, time_step, input_size)),
                        valid_Y:np.reshape(validY[0:valid_size], (-1, time_step, output_size)).astype(np.float),
                        X:np.reshape(trainX[0:batch_size], (-1, time_step, input_size)),
                        Y:np.reshape(trainY[0:batch_size], (-1, time_step, output_size)) })

                    print("\tvalid_loss: " + str(valid_loss_))
                    writer.add_summary(merged_, i)
            writer.close()

def prediction(testX, testY, scalerY):
    def funcMark(d):
        if d >= 0.0015:
            return -1
        elif -0.0015 < d and d < 0.0015:
            return 0
        elif d <= -0.0015:
            return 1

    tf.reset_default_graph()
    X, Y = setup_lstm()
    pred, loss, _ = lstm(1, X, Y)

    saver = tf.train.Saver(tf.global_variables())
    result = []
    with tf.Session() as sess:
        module_file = tf.train.latest_checkpoint(load_modeldir)
        saver.restore(sess, module_file)
        for step in range(400):
            pred_, loss_ = sess.run([pred, loss], feed_dict={
                X:np.reshape(testX[step:step+1], (-1, time_step, input_size)),
                Y:np.reshape(testY[step:step+1], (-1, time_step, output_size)) })
            pred_ = np.reshape(pred_, (-1, len(IDs)))
            truth = np.reshape(testY[step:step+1], (-1, len(IDs)))
            pred_ = scalerY.inverse_transform(pred_)
            truth = scalerY.inverse_transform(truth)
            pred_ = np.reshape(pred_, (time_step, len(IDs)))
            truth = np.reshape(truth, (time_step, len(IDs)))
            d_abs_real = (truth[-1] - truth[-3]) / truth[-3]
            d_abs_pred = (pred_[-1] - pred_[-3]) / truth[-3]
            mark_real = [ funcMark(d) for d in d_abs_real ]
            mark_pred = [ funcMark(d) for d in d_abs_pred ]
            result.append( np.equal(mark_real, mark_pred) )
            print(step, np.equal(mark_real, mark_pred))
            print('\t', mark_real, mark_pred)
    result = np.array(result)
    result = np.sum(result, axis=0)
    result = result / 800
    print(result)




def main(_):
    # util.createDatamat(data_path, IDs)
    dataX, dataY, testX, testY, scalerX, scalerY = readData(data_path, look_back, look_up)

    valid_idxs = random.sample(list(range( len(dataX) )), valid_size)
    validX = np.array([ dataX[idx] for idx in valid_idxs ])
    validY = np.array([ dataY[idx] for idx in valid_idxs ])

    trainX = [ dataX[idx] for idx in range(len(dataX)) if idx not in valid_idxs ]
    trainY = [ dataY[idx] for idx in range(len(dataY)) if idx not in valid_idxs ]

    # train_lstm(trainX, trainY, validX, validY, recover=False)
    prediction(testX, testY, scalerY)


if __name__ == "__main__":
    tf.app.run()
