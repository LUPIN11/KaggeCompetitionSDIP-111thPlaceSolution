import pandas as pd
from sklearn.model_selection import train_test_split


# df_train = pd.read_csv('train.csv')[['prompt', 'filepath']]
# df_0 = pd.read_csv('pis0.csv')[['prompt', 'filepath']]
# df_0_t, df_0_v = train_test_split(df_0, test_size=0.05, random_state=42)
# df_train = pd.concat([df_train, df_0_t], ignore_index=True)
# df_valid = pd.read_csv('valid.csv')[['prompt', 'filepath']]
# df_valid = pd.concat([df_valid, df_0_v], ignore_index=True)
#
# df_train.to_csv('train0.csv')
# df_valid.to_csv('valid0.csv')
# print(len(df_train))
# print(len(df_valid))
# print(df_train.columns)
# print(df_valid.columns)

df_train = pd.read_csv('train0.csv')[['prompt', 'filepath']]
print(len(df_train))
df_0 = pd.read_csv('pis1.csv')[['prompt', 'filepath']]
df_0_t, df_0_v = train_test_split(df_0, test_size=0.05, random_state=42)
df_train = pd.concat([df_train, df_0_t], ignore_index=True)
df_valid = pd.read_csv('valid0.csv')[['prompt', 'filepath']]
print(len(df_valid))
df_valid = pd.concat([df_valid, df_0_v], ignore_index=True)

df_train.to_csv('train1.csv')
df_valid.to_csv('valid1.csv')
print(len(df_train))
print(len(df_valid))
print(df_train.columns)
print(df_valid.columns)