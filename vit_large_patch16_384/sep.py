import pandas as pd
import sys
# df1 = pd.read_csv('dup_valid.csv')[['filepath', 'prompt']].copy()
# df2 = pd.read_csv('valid.csv')[['filepath', 'prompt']].copy()
# df3 = pd.read_csv('train.csv')[['filepath', 'prompt']].copy()
# print(df1.columns)
# print(df2.columns)
# print(df3.columns)
# print(len(df1))
# print(len(df2))
# print(len(df3))


# df1 = pd.read_csv('train.csv')
# # df2 = pd.read_csv('train.csv')
# df3 = df1.sample(n=50000, random_state=42)
# # df4 =
#
# df3.to_csv('sm_train.csv')
# # df4.to_csv()



df_train = pd.read_csv('train.csv')
print(len(df_train))
vector = model.encode(df_train["prompt"].to_numpy(), batch_size=512, show_progress_bar=True, device="cuda")
df_train['simlarity'] = vector
print(df_train.columns)
print(len(df_train))

# df_valid = pd.read_csv('valid.csv')
# print(len(df_valid))
# vector = model.encode(df_valid["prompt"].to_numpy(), batch_size=512, show_progress_bar=True, device="cuda")
# df_valid['simlarity'] = vector
# print(df_valid.columns)
# print(len(df_valid))
