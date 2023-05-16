import pandas as pd
from matplotlib import pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('diffusion.csv')
df['filepath'] = pd.read_csv('train.csv')['filepath']
print(len(df))
print(df.columns)
# plt.hist(df['similarity'], bins=1000, range=(0, 1))
# plt.show()

df_sorted = df.sort_values('similarity')
print(df_sorted.iloc[0]['similarity'])
print(df_sorted.iloc[-1]['similarity'])

bins = pd.interval_range(start=0.0082473754882812, end=0.48828125, freq=0.0001)
counts = pd.cut(df['similarity'], bins=bins, include_lowest=True, right=False).value_counts()
print(counts)

idx = 0
l = len(df)


def get_weight(sim):
    global idx
    print(idx)
    idx += 1
    interval = pd.cut([sim], bins=bins, include_lowest=True, right=False)[0]
    count = counts.get(interval, 0)
    return l / count


df['weight'] = df['similarity'].apply(lambda x: abs(x-0.6)**6)
from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# df['weight'] = scaler.fit_transform(df[['weight']])
# print(df['weight'].sort_values().iloc[490000])

df = df[['prompt', 'filepath', 'weight']]
print(len(df))
print(df.columns)
df.to_csv('weighted_train_s.csv')
