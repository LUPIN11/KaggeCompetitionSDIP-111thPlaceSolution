import pandas as pd

# 读取CSV文件
# df = pd.read_parquet('/root/autodl-tmp/kaggle/img2text/input/metadata.parquet',
#                      columns=['image_name', 'prompt', 'width', 'height'])
# df = pd.read_csv('./diffusiondb_1m.csv')
# # 筛选出具有重复“filepath”值的行
# duplicate_rows = df[df.duplicated(['prompt'], keep=False)]
#
# # 筛选出不重复的行
# unique_rows = df.drop_duplicates(['prompt'], keep=False)
#
# print(len(duplicate_rows))
# print(len(unique_rows))
#
# # # 将重复行和不重复行保存为两个CSV文件
# duplicate_rows.to_csv('diffusiondb_12n.csv', index=False)
# unique_rows.to_csv('diffusiondb_121.csv', index=False)
df1 = pd.read_csv('diffusiondb_121_train.csv')
df2 = pd.read_csv('diffusiondb_121_valid.csv')
print(len(df1))
print(len(df2))

# # 读取CSV文件
# df = pd.read_csv('/root/autodl-tmp/kaggle/img2text/input/diffusiondb_12n.csv')
#
# # 选取前10万行
# new_df = df.iloc[:100000]
#
# # 将新的DataFrame保存为CSV文件
# new_df.to_csv('/root/autodl-tmp/kaggle/img2text/input/diffusiondb_vaild_100k.csv', index=False)
#
# # 删除原来的10万行
# df.drop(df.index[:100000], inplace=True)
#
# # 将剩下的数据保存到原来的CSV文件中
# df.to_csv('/root/autodl-tmp/kaggle/img2text/input/diffusiondb_12n.csv', index=False)
# 读取CSV文件
# df1 = pd.read_csv('/root/autodl-tmp/kaggle/img2text/input/diffusiondb_121.csv')
# df2 = pd.read_csv('')

# # 合并两个数据帧
# merged_df = pd.concat([df1, df2])
#
# # 写入新的CSV文件
# merged_df.to_csv('merged.csv', index=False)


# 读取原始csv文件
# df = pd.read_csv('diffusiondb_121.csv')
#
# # 取前十万行
# df_head = df.iloc[:100000, :]
# # 取其余数据
# df_rest = df.iloc[100000:, :]
#
# # 保存为新的csv文件
# df_head.to_csv('diffusiondb_121_vaild.csv', index=False)
# df_rest.to_csv('diffusiondb_121_train.csv', index=False)
# print(len(df_rest))

#
# # read in the original CSV file
# df = pd.read_csv('./input/diffusiondb_12n.csv')
#
# # sort the DataFrame by the 'prompt' column
# df_sorted = df.sort_values(by='prompt')
# #
# # df_sorted.to_csv('./input/diffusiondb_12n_grouped.csv', index=False)
#
# print(len(df))
# print(len(df_sorted))



# 读取csv文件
# df = pd.read_csv('ddb600k.csv')
# print(len(df))
# df.drop_duplicates(subset='prompt', inplace=True)
# print(len(df))
# df.to_csv('all.csv')
# # 第一次抽取10万行
# df1 = df.sample(n=131724, random_state=42)
# # df1 = df.iloc[:100000, :]
# # df2 = df.iloc[100000:, :]
# #
# # # # 在剩下的70万里抽取20万行
# df2 = df.drop(df1.index)
#
# print(len(df1), len(df2))
# print(df1.columns)
# print(df2.columns)
# df1.to_csv('valid130k.csv', index=False)
# df2.to_csv('train500k.csv', index=False)
#
# # # 获取剩下的50万行
# # df3 = df.drop(df1.index).drop(df2.index)
# print(len(df1))
# print(len(df2))
# # print(len(df3))
# # 将三个数据框保存为csv文件
# df1.to_csv('sm_train.csv', index=False)
# df2.to_csv('sm_valid.csv', index=False)
# # df3.to_csv('diffusiondb_500k.csv', index=False)

