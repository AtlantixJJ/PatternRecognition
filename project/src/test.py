import sys
sys.path.insert(0, "src")
import dataset
import pandas as pd

DATA_DIR = "futuresData"
all_csv = dataset.read_all_data(DATA_DIR)