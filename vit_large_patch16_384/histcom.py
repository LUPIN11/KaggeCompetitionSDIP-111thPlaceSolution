import pandas as pd
import matplotlib.pyplot as plt
import random


# df1 = pd.read_csv('6472_on_diffusiondb_121_train.csv')
# df2 = pd.read_csv('6472_on_diffusiondb_12n.csv')
# df3 = pd.read_csv('6480on_train_1.csv')
# df = pd.concat([df1, df2], ignore_index=True)
# df_sorted = df.sort_values(by='similarity', ascending=True)
# df_hard_o = df_sorted.iloc[:100000, :]
# df_hard_n = df3.loc[df3['prompt'].isin(df_hard_o['prompt'].unique())]
# print(len(df_hard_o))
# print(len(df_hard_n))
# prompt_values = df2['prompt'].unique()
# result_df = df1.loc[df1['prompt'].isin(prompt_values)]
# result_df.drop_duplicates(subset='prompt', inplace=True)
# print(len(result_df))
# # # 绘制直方图
# df1 = pd.read_csv('6480on_train_1.csv')
# df2 = pd.read_csv('6480on_train_2.csv')
# plt.hist(df1['similarity'], bins=1000, alpha=0.5, label='train1', range=(0, 1))
# plt.hist(df2['similarity'], bins=1000, alpha=0.5, label='train2', range=(0, 1))
# plt.legend(loc='upper right')
# plt.title('Similarity Distribution Comparison')
# plt.xlabel('Similarity')
# plt.ylabel('Frequency')
# plt.show()
# print(len(df1[df1['similarity'] < 0.4]))
# print(len(df2[df2['similarity'] < 0.4]))

# df = pd.read_csv('6480on_train_2.csv')
# df_sorted = df.sort_values(by='similarity', ascending=True)
# # print(len(df[df['similarity'] < 0.3]))
# # print(len(df[df['similarity'] < 0.3]['prompt'].unique()))
# # df2 = df_sorted.iloc[:10000, :]
# # print(len(df2['prompt'].unique()))
# df = df[df['similarity'] < 0.3]
# df1 = df.iloc[:5000, :]
# df2 = df.iloc[5000:10000, :]
# df3 = df.iloc[10000:, :]
# print(len(df1))
# print(len(df2))
# print(len(df3))
# df1.to_csv('p5.csv')
# df2.to_csv('p6.csv')
# df3.to_csv('p7.csv')

# df1 = pd.read_csv('./input/gustavosta.csv')
# df2 = pd.read_csv('./input/sd2.csv')
# print(len(df1))
# print(len(df2))
# df = pd.concat([df1, df2], ignore_index=True)
# print(len(df))
# print(df.columns)
# df.to_csv('new_data.csv')