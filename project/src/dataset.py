import os
import tensorflow as tf
import tensorpack as tp

def read_all_text(data_dir):
    files = os.listdir(data_dir)
    