import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
plt.plot(range(10))
plt.close()

import pandas as pd
import src.dataset as dataset

FILE_NAME = "futuresData/0-20170703-day_A1.csv"

df = pd.read_csv(FILE_NAME, index_col=0, parse_dates=True)
df[u'meanPrice'] = dataset.get_mean_price(df[u'bidPrice1'])