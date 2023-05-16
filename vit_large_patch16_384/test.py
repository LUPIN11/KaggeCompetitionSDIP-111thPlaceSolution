import pandas as pd
from matplotlib import pyplot as plt
# df1 = pd.read_csv('6480on_train_1.csv')
# df2 = pd.read_csv('new1.csv')
# df3 = pd.read_csv('new2.csv')
# df4 = pd.concat([df2, df3], ignore_index=True)
# df_sorted = df1.sort_values(by='similarity', ascending=True)
# df_min_similarity = df_sorted.head(10000)
# print(len(df4['prompt'].unique()))
# print(len(df_min_similarity['prompt'].unique()))
# num_of_duplicates = len(set(df4['prompt']).intersection(set(df_min_similarity['prompt'])))
# print(num_of_duplicates)
# df = pd.read_csv('6472_on_diffusiondb_121_train.csv')
# df1 = pd.read_csv('6480on_train_1.csv')
# df2 = pd.read_csv('6480on_train_2.csv')
# df1 = df1.sort_values(by='similarity', ascending=True)
# prompts = df1.iloc[:10000, :]['prompt'].unique()


# # prompts = df['prompt'].unique()
# # df = df[df['prompt'].isin(prompts)]
# df1 = df1[df1['prompt'].isin(prompts)]
# df2 = df2[df2['prompt'].isin(prompts)]
# print(len(df))
# print(len(df1))
# print(len(df2))
# # plt.hist(df['similarity'], bins=100, alpha=0.4, label='origin', range=(0, 1))
# plt.hist(df1['similarity'], bins=100, alpha=0.4, label='train_1', range=(0, 1))
# plt.hist(df2['similarity'], bins=100, alpha=0.4, label='train_2', range=(0, 1))
# plt.legend(loc='upper right')
# plt.title('Similarity Distribution Comparison')
# plt.xlabel('Similarity')
# plt.ylabel('Frequency')
# plt.show()


# print(len(df))
# df_sorted = df.sort_values(by='similarity', ascending=True)
# plt.hist(df_sorted['similarity'], bins=1000, alpha=0.5, label='train_2', range=(0, 1))
# plt.legend(loc='upper right')
# plt.title('Similarity Distribution Comparison')
# plt.xlabel('Similarity')
# plt.ylabel('Frequency')
# plt.show()
# print(df['similarity'].mean())

# print(len(df2))
# print(df2.columns)
# df1 = pd.read_csv('6480on_new_data.csv')
# df2 = df1[df1['similarity'] < 0.5]
# df3 = pd.read_csv('new5.csv')
# df4 = pd.read_csv('new6.csv')
# df5 = pd.read_csv('new7.csv')
# df6 = pd.read_csv('train_2.csv')
# df = pd.concat([df2, df3, df4, df5, df6], ignore_index=True)
# df = df[['prompt', 'filepath']].copy()
# print(len(df1))
# print(len(df2))
# print(len(df3))
# print(len(df4))
# print(len(df5))
# print(len(df6))
# print(len(df))
# print(df.columns)
# df.to_csv('train_3.csv')

# df = pd.read_csv('6473on_train_3.csv')
# df1 = df[(df['similarity'] > 0.35) & (df['similarity'] < 0.45)]
# df2 = df1.iloc[:10000, :]
# df3 = df1.iloc[10000:20000, :]
# df4 = df1.iloc[20000:30000, :]
# df5 = df1.iloc[30000:40000, :]
# print(len(df1))
# print(len(df2))
# print(len(df3))
# print(len(df4))
# print(len(df5))
# print(len(df))
# df2.to_csv('p8.csv')
# df3.to_csv('p9.csv')
# df4.to_csv('p10.csv')
# df5.to_csv('p11.csv')

df8 = pd.read_csv('new8.csv')
df9 = pd.read_csv('new9.csv')
df10 = pd.read_csv('new10.csv')
df11 = pd.read_csv('new11.csv')
df_n3 = pd.read_csv('train_3.csv')
df = pd.concat([df8, df9, df10, df11, df_n3], ignore_index=True)
df = df[['prompt','filepath']].copy()
print(len(df))
print(df.columns)
df.to_csv('train_4.csv')
