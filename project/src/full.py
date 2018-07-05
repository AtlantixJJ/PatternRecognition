import tensorflow as tf
import torch
import sys
sys.path.insert(0, "src")
import dataset, lib
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import tensorpack as tp
from os.path import join as pj
import tqdm, pprint

TLEN = dataset.TLEN
INST_TYPE = lib.INST_TYPE
INPUT_LEN = dataset.INPUT_LEN
OUTPUT_LEN = dataset.OUTPUT_LEN
#BATCH_SIZE = 256; NUM_WORKER = 32
BATCH_SIZE = 4; NUM_WORKER = 4
N_EPOCH = 10

STAIRCASE = 2
EPSILON = 1e-6

tmp_x = np.array(range(0, OUTPUT_LEN))
tmp_x2 = tmp_x ** 2
sum_tmp_x = tmp_x.sum()
sum_tmp_x2 = tmp_x2.sum()
tmp_div = OUTPUT_LEN * sum_tmp_x2 - sum_tmp_x ** 2

#DATA_DIR = "futuresData"
#futures_data = dataset.FuturesData(DATA_DIR)
#futures_data.save_npz()

def analysis(est_label, gt_label):
    print(est_label.shape, gt_label.shape)

    recalls = []
    precisions = []

    for i in range(1, 6):
        pred_i = (est_label == i).astype("int32")
        gt_i = (gt_label == i).astype("int32")
        pred_total = pred_i.sum()
        gt_total = gt_i.sum()
        match_total = (pred_i * gt_i).sum()

        if pred_total == 0:
            recall_i = 0.0
        else:
            recall_i = float(match_total) / pred_total
        
        if gt_total == 0:
            precision_i = 0.0
        else:
            precision_i = float(match_total) / gt_total

        recalls.append(recall_i)
        precisions.append(precision_i)

        print("Recall:%.5f,Precision:%.5f" % (recall_i, precision_i))
    
    match_total = (est_label == gt_label).astype("int32").sum()
    precision = float(match_total) / est_label.shape[0]

    print("Precision:%.5f" % (precision))
    return precision, recalls, precisions

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
    #dl.reset_state()

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
    tp.logger.info("Test combine")
    futures_dataset = dataset.FuturesData(is_train=True, from_npz=True)
    # build dataset
    train_dl = DataLoader(futures_dataset, BATCH_SIZE, num_workers=NUM_WORKER)
    val_dl = DataLoader(futures_dataset, BATCH_SIZE, num_workers=NUM_WORKER, shuffle=False)

    # control var
    lr = tf.placeholder(tf.float32, [], "lr")

    # build network
    x = tf.placeholder(tf.float32, [None, 4, INPUT_LEN], name="x")
    y = tf.placeholder(tf.float32, [None, 4, OUTPUT_LEN], name="x")
    est_y = mlp_combine(x)

    regression_loss_inst = tf.reduce_mean(tf.abs(est_y - y), axis=[0, 2])
    regression_loss = tf.reduce_mean(regression_loss_inst)
    optim = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(regression_loss)

    inst_summary = [tf.summary.scalar("regression/train/inst%s" % INST_TYPE[i], regression_loss_inst[i]) for i in range(4)]
    train_summary = tf.summary.merge(inst_summary)
    summary_writer = tf.summary.FileWriter("logs/" + name)

    saver = tf.train.Saver()

    # init
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        global_iter = 0
        for epoch_id in range(N_EPOCH):
            LR = 0.1 ** (N_EPOCH // STAIRCASE)
            
            train_dl.dataset.is_train = True
            for sample in tqdm.tqdm(train_dl):
                train_seq = sample['seq'].numpy()
                target_seq = sample['target'].numpy()

                if target_seq.shape[2] != OUTPUT_LEN:
                    print(target_seq.shape)
                    continue

                sum_, _ = sess.run([train_summary, optim], {x: train_seq, y: target_seq, lr: LR})
                summary_writer.add_summary(sum_, global_iter)
                global_iter += 1

            # four inst type loss together
            tot_reg_loss = 0
            gt_label = []
            est_label = []
            sample_cnt = 0

            tp.logger.info("Validation")
            #val_dl.reset_state()
            val_dl.dataset.is_train = False
            for idx, sample in enumerate(val_dl):
                train_seq = sample['seq'].numpy()
                target_seq = sample['target'].numpy()
                label_seq = sample['label'].numpy()

                if target_seq.shape[2] != OUTPUT_LEN:
                    print(target_seq.shape)
                    continue

                reg_loss_int, pred = sess.run([regression_loss_inst, est_y],
                                    {
                                        x: train_seq,
                                        y: target_seq
                                    })
                
                # get class
                for i in range(pred.shape[0]):
                    local = []
                    for inst_type in range(4):
                        k = (OUTPUT_LEN * (tmp_x * pred[i, inst_type]).sum() - sum_tmp_x * pred[i, inst_type].sum()) / tmp_div / 5000.0
                        local.append(lib.get_class(k))
                    est_label.append(local)
                gt_label.append(label_seq)
                
                # record an average
                tot_reg_loss += reg_loss_int * train_seq.shape[0] / float(BATCH_SIZE)

                sample_cnt += 1


            tp.logger.info("Validation done")
            est_label = np.array(est_label)
            gt_label = np.concatenate(gt_label)
            # get statistics
            for inst_type in range(4):
                precision, recalls, precisions = analysis(
                    est_label[:, inst_type],
                    gt_label[:, inst_type])

                summary = tf.Summary(value=[
                    tf.Summary.Value(tag="regression/val/inst%s" % INST_TYPE[inst_type], simple_value=tot_reg_loss[inst_type]/sample_cnt),
                    tf.Summary.Value(tag="precision/val/inst%s" % INST_TYPE[inst_type], simple_value=precision)]
                    )
                summary_writer.add_summary(summary, epoch_id)

                summary = tf.Summary(value=[tf.Summary.Value(
                        tag="precision%d/val/inst%s" % (cls_id, INST_TYPE[inst_type]),
                        simple_value=precisions[cls_id-1]) for cls_id in range(1, 6)
                    ])
                summary_writer.add_summary(summary, epoch_id)

                summary = tf.Summary(value=[tf.Summary.Value(
                        tag="recall%d/val/inst%s" % (cls_id, INST_TYPE[inst_type]),
                        simple_value=recalls[cls_id-1]) for cls_id in range(1, 6)
                    ])
                summary_writer.add_summary(summary, epoch_id)

            save_path = saver.save(sess, pj("model", name + ".ckpt"))
            print("Model saved in %s" % save_path)

def test_single(is_linear=True, name="single"):
    tp.logger.info("Test linear " + name)
    futures_dataset = dataset.FuturesData(is_train=True, from_npz=True)
    # build dataset
    train_dl = DataLoader(futures_dataset, BATCH_SIZE, num_workers=NUM_WORKER)
    val_dl = DataLoader(futures_dataset, BATCH_SIZE, num_workers=NUM_WORKER, shuffle=False)

    # control var
    lr = tf.placeholder(tf.float32, [], "lr")

    # build network
    x = tf.placeholder(tf.float32, [None, INPUT_LEN], name="x")
    y = tf.placeholder(tf.float32, [None, OUTPUT_LEN], name="x")

    summary_writer = tf.summary.FileWriter("logs/" + name)

    def run(inst_type):
        global_iter = 0

        if is_linear:
            est_y = linear(x)
        else:
            est_y = mlp_single(x)
        
        saver = tf.train.Saver()
        regression_loss = tf.reduce_mean(tf.abs(est_y - y))
        optim = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(regression_loss)
        sess.run(tf.global_variables_initializer())

        inst_summary = tf.summary.scalar("regression/train/inst%s" % INST_TYPE[inst_type], regression_loss)

        for epoch_id in range(N_EPOCH):
            tp.logger.info("Epoch %d" % epoch_id)
            LR = 0.1 ** (N_EPOCH // STAIRCASE)
            
            train_dl.dataset.is_train = True
            for sample in tqdm.tqdm(train_dl):
                train_seq = sample['seq'].numpy()
                target_seq = sample['target'].numpy()
                
                if target_seq.shape[2] != OUTPUT_LEN:
                    print(target_seq.shape)
                    continue

                sum_, _ = sess.run([inst_summary, optim],
                                    {
                                        x: train_seq[:, inst_type, :],
                                        y: target_seq[:, inst_type, :],
                                        lr: LR
                                    })
                summary_writer.add_summary(sum_, global_iter)
                global_iter += 1
            
            tot_reg_loss = 0
            sample_cnt = 0

            est_label = []
            gt_label = []
            tp.logger.info("Validation")
            #val_dl.reset_state()
            val_dl.dataset.is_train = False
            for sample in tqdm.tqdm(val_dl):
                train_seq = sample['seq'].numpy()
                target_seq = sample['target'].numpy()
                label_seq = sample['label'].numpy()

                reg_loss, pred = sess.run([regression_loss, est_y],
                                    {
                                        x: train_seq[:, inst_type, :],
                                        y: target_seq[:, inst_type, :]
                                    })
                
                # get class
                for i in range(pred.shape[0]):
                    k = (OUTPUT_LEN * (tmp_x * pred[i]).sum() - sum_tmp_x * pred[i].sum()) / tmp_div / 5000.0
                    est_label.append(lib.get_class(k))
                gt_label.append(label_seq[:, inst_type])
                
                # record an average
                tot_reg_loss += reg_loss * train_seq.shape[0] / float(BATCH_SIZE)

                sample_cnt += 1
            
            tp.logger.info("Validation done")

            # get statistics
            precision, recalls, precisions = analysis(np.array(est_label), np.concatenate(gt_label))

            summary = tf.Summary(value=[
                tf.Summary.Value(tag="regression/val/inst%s" % INST_TYPE[inst_type], simple_value=tot_reg_loss/sample_cnt),
                tf.Summary.Value(tag="precision/val/inst%s" % INST_TYPE[inst_type], simple_value=precision)
                ])
            summary_writer.add_summary(summary, epoch_id)

            summary = tf.Summary(value=[tf.Summary.Value(
                    tag="precision%d/val/inst%s" % (cls_id, INST_TYPE[inst_type]),
                    simple_value=precisions[cls_id-1]) for cls_id in range(1, 6)
                ])
            summary_writer.add_summary(summary, epoch_id)

            summary = tf.Summary(value=[tf.Summary.Value(
                    tag="recall%d/val/inst%s" % (cls_id, INST_TYPE[inst_type]),
                    simple_value=recalls[cls_id-1]) for cls_id in range(1, 6)
                ])
            summary_writer.add_summary(summary, epoch_id)

            save_path = saver.save(sess, pj("model", name + "_" + str(inst_type) + ".ckpt"))
            tp.logger.info("Model saved in %s" % save_path)
    # init
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        for inst_type in range(4):
            run(inst_type)

def run_all_model():
    tp.logger.info("Test all ")
    futures_dataset = dataset.FuturesData(is_train=True, from_npz=True)

    # build dataset
    train_dl = DataLoader(futures_dataset, BATCH_SIZE, num_workers=NUM_WORKER)
    val_dl = DataLoader(futures_dataset, BATCH_SIZE, num_workers=NUM_WORKER, shuffle=False)

    # control
    lr = tf.placeholder(tf.float32, [], "lr")

    #x_inst = [tf.placeholder(tf.float32, [None, INPUT_LEN], name="x_%d"%i) for i in range(4)]
    #y_inst = [tf.placeholder(tf.float32, [None, OUTPUT_LEN], name="x_%d"%i) for i in range(4)]
    x = tf.placeholder(tf.float32, [None, 4, INPUT_LEN], name="x")
    y = tf.placeholder(tf.float32, [None, 4, OUTPUT_LEN], name="x")

    # get prediction
    with tf.variable_scope("combine"):
        est_combine_y = mlp_combine(x)
        combine_var = [v for v in tf.trainable_variables() if "combine" in v.name]
    est_linear_y = []
    est_mlp_y = []
    for i in range(4):
        with tf.variable_scope("inst%s"%INST_TYPE[i]):
            with tf.variable_scope("linear"):
                est_linear_y.append(linear(x[:, i]))
            with tf.variable_scope("mlp"):
                est_mlp_y.append(mlp_single(x[:, i]))
    
    # get loss
    regression_loss_combine_inst = tf.reduce_mean(tf.abs(est_combine_y - y), axis=[0, 2])
    regression_loss_combine = tf.reduce_mean(regression_loss_combine_inst)
    combine_optim = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(regression_loss_combine, var_list=combine_var)
    tp.logger.info("Combine MLP variables:")
    pprint.pprint(combine_var)

    # get training loss
    regression_loss_linear_inst = []
    linear_optim_inst = []
    linear_var = []
    regression_loss_mlp_inst = []
    mlp_optim_inst = []
    mlp_var = []
    for i in range(4):
        regression_loss_linear_inst.append(tf.reduce_mean(tf.abs(est_linear_y[i] - y[:, i])))
        regression_loss_mlp_inst.append(tf.reduce_mean(tf.abs(est_mlp_y[i] - y[:, i])))

        linear_var.append([v for v in tf.trainable_variables() if "linear" in v.name and "inst%s"%INST_TYPE[i] in v.name])
        mlp_var.append([v for v in tf.trainable_variables() if "mlp" in v.name and "inst%s"%INST_TYPE[i] in v.name])

        linear_optim_inst.append(tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(regression_loss_linear_inst[i], var_list=linear_var[i]))
        mlp_optim_inst.append(tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(regression_loss_mlp_inst[i], var_list=mlp_var[i]))
        tp.logger.info("linear variables (%s):" % INST_TYPE[i])
        pprint.pprint(linear_var[i])
        tp.logger.info("Single MLP variables (%s):" % INST_TYPE[i])
        pprint.pprint(mlp_var[i])
    train_optim_collections = [combine_optim] + linear_optim_inst + mlp_optim_inst

    # get training summary
    summary_writers = []
    summary_writers.append(tf.summary.FileWriter("logs/combine_mlp"))
    summary_writers.append(tf.summary.FileWriter("logs/linear_mlp"))
    summary_writers.append(tf.summary.FileWriter("logs/single_mlp"))

    inst_summary_combine = []
    inst_summary_linear  = []
    inst_summary_mlp     = []
    with tf.variable_scope("regression", reuse=True):
        for inst_type in range(4):
            inst_summary_combine.append(tf.summary.scalar(
                "/train/inst%s" % INST_TYPE[inst_type],
                regression_loss_combine_inst[inst_type]))
            inst_summary_linear.append(tf.summary.scalar(
                "/train/inst%s" % INST_TYPE[inst_type],
                regression_loss_linear_inst[inst_type]))
            inst_summary_mlp.append(tf.summary.scalar(
                "/train/inst%s" % INST_TYPE[inst_type],
                regression_loss_mlp_inst[inst_type]))
    inst_summary_combine = tf.summary.merge(inst_summary_combine )
    inst_summary_linear  = tf.summary.merge(inst_summary_linear  )
    inst_summary_mlp     = tf.summary.merge(inst_summary_mlp     )
    train_summary_collections = [inst_summary_combine, inst_summary_linear, inst_summary_mlp]

    saver = tf.train.Saver()

    # init
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        global_iter = 0
        for epoch_id in range(N_EPOCH):
            LR = 0.1 ** (N_EPOCH // STAIRCASE)
            
            train_dl.dataset.is_train = True
            for sample in tqdm.tqdm(train_dl):
                train_seq = sample['seq'].numpy()
                target_seq = sample['target'].numpy()

                result = sess.run(train_summary_collections + train_optim_collections, {x: train_seq, y: target_seq, lr: LR})
                for sum_id in range(3):
                    summary_writers[sum_id].add_summary(result[sum_id], global_iter)
                    tf.reset_default_graph()
                    summary_writers[sum_id].close()
                for sum_id in range(3):
                    summary_writers[sum_id].reopen()
                global_iter += 1

            # four inst type loss together
            tot_reg_loss = [0.0, 0.0, 0.0]
            gt_label = []
            est_label = []
            sample_cnt = 0

            tp.logger.info("Validation")
            #val_dl.reset_state()
            val_dl.dataset.is_train = False
            for idx, sample in enumerate(val_dl):
                train_seq = sample['seq'].numpy()
                target_seq = sample['target'].numpy()
                label_seq = sample['label'].numpy()

                reg_loss_int, pred = sess.run([regression_loss_inst, est_y],
                                    {
                                        x: train_seq,
                                        y: target_seq
                                    })
                
                # get class
                for i in range(pred.shape[0]):
                    local = []
                    for inst_type in range(4):
                        k = (OUTPUT_LEN * (tmp_x * pred[i, inst_type]).sum() - sum_tmp_x * pred[i, inst_type].sum()) / tmp_div / 5000.0
                        local.append(lib.get_class(k))
                    est_label.append(local)
                gt_label.append(label_seq)
                
                # record an average
                tot_reg_loss += reg_loss_int * train_seq.shape[0] / float(BATCH_SIZE)

                sample_cnt += 1

            save_path = saver.save(sess, pj("model", "all_model.ckpt"))
            print("Model saved in %s" % save_path)

def run_single_model(is_linear, name):
    tp.logger.info("Test all single model")
    futures_dataset = dataset.FuturesData(is_train=True, from_npz=True)

    # build dataset
    train_dl = DataLoader(futures_dataset, BATCH_SIZE, num_workers=NUM_WORKER)
    val_dl = DataLoader(futures_dataset, BATCH_SIZE, num_workers=NUM_WORKER, shuffle=False)

    # control
    lr = tf.placeholder(tf.float32, [], "lr")

    #x_inst = [tf.placeholder(tf.float32, [None, INPUT_LEN], name="x_%d"%i) for i in range(4)]
    #y_inst = [tf.placeholder(tf.float32, [None, OUTPUT_LEN], name="x_%d"%i) for i in range(4)]
    x = tf.placeholder(tf.float32, [None, 4, INPUT_LEN], name="x")
    y = tf.placeholder(tf.float32, [None, 4, OUTPUT_LEN], name="x")

    if is_linear:
        net_func = linear
    else:
        net_func = mlp_single

    est_y = []
    for i in range(4):
        with tf.variable_scope("inst%s"%INST_TYPE[i]):
            est_y.append(net_func(x[:, i]))

    # get training loss
    regression_loss_inst = []
    optim_inst = []
    var = []

    for i in range(4):
        regression_loss_inst.append(tf.reduce_mean(tf.abs(est_y[i] - y[:, i])))
        var.append([v for v in tf.trainable_variables() if "inst%s"%INST_TYPE[i] in v.name])

        optim_inst.append(tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(regression_loss_inst[i], var_list=var[i]))
        tp.logger.info("variables (%s):" % INST_TYPE[i])
        pprint.pprint(var[i])

    # get training summary
    summary_writer = tf.summary.FileWriter(pj("logs/", name))
    inst_summary = []
    with tf.variable_scope("regression", reuse=True):
        for inst_type in range(4):
            inst_summary.append(tf.summary.scalar(
                "/train/inst%s" % INST_TYPE[inst_type],
                regression_loss_inst[inst_type]))
    train_summary = tf.summary.merge(inst_summary)

    saver = tf.train.Saver()

    # init
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        global_iter = 0
        for epoch_id in range(N_EPOCH):
            LR = 0.1 ** (N_EPOCH // STAIRCASE)
            
            train_dl.dataset.is_train = True
            for sample in tqdm.tqdm(train_dl):
                train_seq = sample['seq'].numpy()
                target_seq = sample['target'].numpy()

                result = sess.run([train_summary] + optim_inst, {x: train_seq, y: target_seq, lr: LR})
                summary_writer.add_summary(result[0], global_iter)
                global_iter += 1

            # four inst type loss together
            tot_reg_loss = 0
            gt_label = []
            est_label = []
            sample_cnt = 0

            tp.logger.info("Validation")
            val_dl.dataset.is_train = False
            for idx, sample in enumerate(val_dl):
                train_seq = sample['seq'].numpy()
                target_seq = sample['target'].numpy()
                label_seq = sample['label'].numpy()

                if target_seq.shape[2] != OUTPUT_LEN:
                    print(target_seq.shape)
                    continue

                result = sess.run(regression_loss_inst + est_y,
                                    {
                                        x: train_seq,
                                        y: target_seq
                                    })
                
                pred = np.stack(result[-4:], 1)
                reg_loss_int = np.stack(result[:4])

                # get class
                for i in range(pred.shape[0]):
                    local = []
                    for inst_type in range(4):
                        k = (OUTPUT_LEN * (tmp_x * pred[i, inst_type]).sum() - sum_tmp_x * pred[i, inst_type].sum()) / tmp_div / 5000.0
                        local.append(lib.get_class(k))
                    est_label.append(local)
                gt_label.append(label_seq)
                
                # record an average
                tot_reg_loss += reg_loss_int * train_seq.shape[0] / float(BATCH_SIZE)

                sample_cnt += 1

            tp.logger.info("Validation done")
            est_label = np.array(est_label)
            gt_label = np.concatenate(gt_label)
            # get statistics
            for inst_type in range(4):
                precision, recalls, precisions = analysis(
                    est_label[:, inst_type],
                    gt_label[:, inst_type])

                summary = tf.Summary(value=[
                    tf.Summary.Value(tag="regression/val/inst%s" % INST_TYPE[inst_type], simple_value=tot_reg_loss[inst_type]/sample_cnt),
                    tf.Summary.Value(tag="precision/val/inst%s" % INST_TYPE[inst_type], simple_value=precision)]
                    )
                summary_writer.add_summary(summary, epoch_id)

                summary = tf.Summary(value=[tf.Summary.Value(
                        tag="precision%d/val/inst%s" % (cls_id, INST_TYPE[inst_type]),
                        simple_value=precisions[cls_id-1]) for cls_id in range(1, 6)
                    ])
                summary_writer.add_summary(summary, epoch_id)

                summary = tf.Summary(value=[tf.Summary.Value(
                        tag="recall%d/val/inst%s" % (cls_id, INST_TYPE[inst_type]),
                        simple_value=recalls[cls_id-1]) for cls_id in range(1, 6)
                    ])
                summary_writer.add_summary(summary, epoch_id)

            save_path = saver.save(sess, pj("model", "all_model.ckpt"))
            print("Model saved in %s" % save_path)


if __name__ == "__main__":
    if sys.argv[1] == "1":
        test_single(True, "single_linear")
    elif sys.argv[1] == "2":
        test_single(False, "single_mlp")
    elif sys.argv[1] == "3":
        test_combine("combine_mlp")
    elif sys.argv[1] == "4":
        run_single_model(True, "single_linear")
    elif sys.argv[1] == "5":
        run_single_model(False, "single_mlp")   
    #run_all_model()
    