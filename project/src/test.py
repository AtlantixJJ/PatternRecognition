import sys
sys.path.insert(0, "src")
import dataset
import numpy as np
import pandas as pd

#DATA_DIR = "futuresData"
#futures_data = dataset.FuturesData(DATA_DIR)
#futures_data.save_npz()

dl = dataset.DataLoader([dataset.FuturesData(from_npz=True)], 64)
for idx, sample in enumerate(dl.generator()):
    sample = np.array(sample[0])
    print(sample.shape)
dl.reset_state()