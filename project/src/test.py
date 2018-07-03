import sys
sys.path.insert(0, "src")
import dataset
import pandas as pd

DATA_DIR = "futuresData"
futures_data = dataset.FuturesData(DATA_DIR)