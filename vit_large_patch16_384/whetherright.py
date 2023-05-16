import pandas as pd
from matplotlib import pyplot as plt

# # 对初始train set中所有样本预测结果的改变
# df_org = pd.read_csv('6472_on_diffusiondb_121_train.csv')
# prompts_org = df_org['prompt'].unique()
# # stage 1
# df_s1 = pd.read_csv('6480on_train_1.csv')
# df_s1_in_org = df_s1[df_s1['prompt'].isin(prompts_org)].copy()
# df_s1_in_org.drop_duplicates(subset='prompt', inplace=True)
# # stage 2
# df_s2 = pd.read_csv('6480on_train_2.csv')
# df_s2_in_org = df_s2[df_s2['prompt'].isin(prompts_org)].copy()
# df_s2_in_org.drop_duplicates(subset='prompt', inplace=True)
# # stage 3
# df_s3 = pd.read_csv('6473on_train_3.csv')
# df_s3_in_org = df_s3[df_s3['prompt'].isin(prompts_org)].copy()
# df_s3_in_org.drop_duplicates(subset='prompt', inplace=True)
# # 分析
# print(len(prompts_org))
# print(len(df_s1_in_org))
# print(len(df_s2_in_org))
# print(len(df_s3_in_org))
# plt.hist(df_org['similarity'], bins=500, alpha=0.5, label='0', range=(0, 1))
# # plt.hist(df_s1_in_org['similarity'], bins=500, alpha=0.5, label='1', range=(0, 1))
# # plt.hist(df_s2_in_org['similarity'], bins=500, alpha=0.3, label='2', range=(0, 1))
# plt.hist(df_s3_in_org['similarity'], bins=500, alpha=0.3, label='3', range=(0, 1))
# plt.legend(loc='upper right')
# plt.title('Similarity Distribution Comparison')
# plt.xlabel('Similarity')
# plt.ylabel('Frequency')
# plt.show()

# 分析2
df_org = pd.read_csv('6472_on_diffusiondb_121_train.csv')
df_org_hard = df_org.sort_values(by='similarity', ascending=True).iloc[:, :]  # 取1万是因为从orn到1就是取了10K
prompts_hard_on_org = df_org_hard['prompt'].unique()
# stage 1
df_s1 = pd.read_csv('6480on_train_1.csv')
df_s1_in_org = df_s1[df_s1['prompt'].isin(prompts_hard_on_org)].copy()
df_s1_in_org.drop_duplicates(subset='prompt', inplace=True)
# stage 2
df_s2 = pd.read_csv('6480on_train_2.csv')
df_s2_in_org = df_s2[df_s2['prompt'].isin(prompts_hard_on_org)].copy()
df_s2_in_org.drop_duplicates(subset='prompt', inplace=True)
# stage 3
df_s3 = pd.read_csv('6473on_train_3.csv')
df_s3_in_org = df_s3[df_s3['prompt'].isin(prompts_hard_on_org)].copy()
df_s3_in_org.drop_duplicates(subset='prompt', inplace=True)
# merge
merged_df = pd.merge(df_s3_in_org, df_org_hard, on='prompt')
merged_df['similarity_diff'] = merged_df['similarity_x'] - merged_df['similarity_y']
# 分析
print(len(prompts_hard_on_org))
print(len(df_s1_in_org))
print(len(df_s2_in_org))
print(len(df_s3_in_org))
print(merged_df['similarity_diff'].mean())
plt.hist(merged_df['similarity_diff'], bins=500, alpha=0.5, label='0', range=(-1, 1))
# plt.hist(df_org_hard['similarity'], bins=500, alpha=0.5, label='0', range=(0, 1))
# plt.hist(df_s1_in_org['similarity'], bins=500, alpha=0.5, label='1', range=(0, 1))
# plt.hist(df_s2_in_org['similarity'], bins=500, alpha=0.5, label='2', range=(0, 1))
# plt.hist(df_s3_in_org['similarity'], bins=500, alpha=0.5, label='3', range=(0, 1))
plt.legend(loc='upper right')
plt.title('Similarity Distribution Comparison')
plt.xlabel('Similarity')
plt.ylabel('Frequency')
plt.show()
"""
分析1结论：
出乎意料的是，在初始训练集上，不同的模型的平均train_cos居然是一样的
分析2结论：
分析了初始train set上最差的10k样本在后续的变化
可以看到，在第一轮之后这些样本的得分有了显著提升，但是第二轮开始到以后，提升都不显著
分析了初始train set上最差的10k样本在后续的变化
居然是同样的
分析了表现最好的40k，结果是这些样本的表现轻微变差
结论：
以上现象说明了，对初始样本来说，我所做的数据增广起的作用是：
好的样本表现变差，坏的样本表现变好了，总体而言结果是相抵消的
这种鱼与熊掌不可得兼的原因可能是两个：
1.模型训练不足（以上结果都是只训练了一个epoch）
2.模型拟合能力不足，应该解封更多的可训练参数才对
"""
"""
另一个现象是在新的train set上，模型的第一轮train-cos不断下降，但是valid cos没有下降，反而有小幅度提升
这是否说明了模型的潜力在提升呢，还是说模型的train-cos其实是总体上降低了也就是train cos上限降低了，而模型的valid上限没改变
如果模型的train cos遇到瓶颈了应该解封更多参数！
"""
# df = pd.read_csv('6473on_train_3.csv')
# df1 = df[df['prompt'].map(lambda x: len(x.split()) < 30)]
# print(len(df1))
# plt.hist(df1['similarity'], bins=500, alpha=0.5, label='over long', range=(0, 1))
# plt.legend(loc='upper right')
# plt.title('Similarity Distribution Comparison')
# plt.xlabel('Similarity')
# plt.ylabel('Frequency')
# plt.show()
"""
神奇的是样本长度好像和学习的难易没有关系
"""
"""
经过分析得知，这几个轮次的变化是：
虽然原来表现差的样本上的效果变好了，但是相应的原来表现好的样本的效果变差了
两种样本双向奔赴，导致总体得分没有变化
但是这种变化必然会在lb上有所体现
比如lb数据集或者真实数据集上，原来表现好的那一类样本占多数，不好的占少数
这样一来，在当前数据集上互补的提升其实是一种degenerate
唯一能想到的解决方法是解封更多层
当前的训练也反应了这个问题，在第三轮次就卡住了，虽然train cos已经比较高了，但是
上个epoch还没过拟合，这轮就卡住了，也说明模型拟合能力不足，甚至可能观察不到过拟合的出现
"""
