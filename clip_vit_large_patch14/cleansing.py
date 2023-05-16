import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

dis = np.load('distance.npy')
# plt.hist(dis, bins=100)
# plt.show()
idx = np.where((dis >7.2)&(dis < 8.7))[0]
print(len(idx))
df = pd.read_csv('train.csv')
df = df.loc[idx]
print(len(df))
df.to_csv('train350k.csv')
