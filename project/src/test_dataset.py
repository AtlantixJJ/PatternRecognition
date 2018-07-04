import sys
sys.path.insert(0, "src")
import dataset, lib
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorpack as tp

INST_TYPE = lib.INST_TYPE
INPUT_LEN = dataset.INPUT_LEN
OUTPUT_LEN = dataset.OUTPUT_LEN
BATCH_SIZE = 64
N_EPOCH = 100
EPSILON = 1e-6

d = dataset.FuturesData(from_npz=True)
