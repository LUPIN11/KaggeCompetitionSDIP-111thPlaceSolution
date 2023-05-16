import matplotlib.pyplot as plt
import pandas as pd

# print(df['similarity'].mean())
# df = df[(df['similarity'] < 0.6) & (df['similarity'] > 0)].copy()
# print(len(df))
# df.to_csv('hard_prompts_from_sd2.csv')
# df = pd.read_csv('resultOnTrain_mse.csv')
# df1 = pd.read_csv('resultOnTrainSet.csv')
# plt.hist(df['similarity'], bins=2000, range=(-1, 1), alpha=0.5)
# plt.hist(df1['similarity'], bins=2000, range=(-1, 1), alpha=0.5)
# plt.title('Similarities Distribution')
# plt.show()

# print(len(df[df['similarity'] < 0.3]))
# df_r = df[(df['similarity'] < 0.4) & (df['similarity'] > 0)]
# df_0 = df_r.sample(n=2000, random_state=42)
# df_0.to_csv('test.csv')
# df = df[df['prompt'].map(lambda x: len(x.split()) < 20)]
# print(len(df_r))
# df = df[df['prompt'].map(lambda x: len(x.split()) < 20)]
# plt.hist(df['similarity'], bins=1000, range=(0, 1))
# plt.title('Similarities Distribution')
# plt.show()

# df = pd.read_csv('resultOnTrainSet.csv')[['prompt', 'similarity']]
# df = df[df['similarity'] > 0]
# print(len(df))
# print(df.columns)
# agg = len(df)
# 统计各个得分段的样本数
# 每0.05为一段
# distrib = [0 for _ in range(20)]
# for i in range(0, 20, 1):
#     print(i / 20, (i + 1) / 20)
#     distrib[i] = len(df[(df['similarity'] >= i / 20) & (df['similarity'] <= (i + 1) / 20)])
#     print(distrib[i])
# print(sum(distrib))
# # distrib = [num/agg for num in distrib]
# # print(distrib)
# # print(sum(distrib))
# weights = [agg/(20*num) for num in distrib]
# print(weights)
# df_sorted = df.sort_values(by='similarity', ascending=True)
# new_col = []
# for i in range(20):
#     new_col.extend([weights[i] for _ in range(distrib[i])])
# df_sorted['weight'] = new_col
# print(df_sorted['similarity'].head())
# # df_sorted.to_csv('weightedResultOnTrain.csv')

# df = pd.read_csv('../fliter/train1.csv')
# print(len(df))
# df.dropna(subset=['filepath'], inplace=True)
# print(len(df))
# print(df.columns)
# df = df[['prompt', 'filepath']]
# df.to_csv('../fliter/train1.csv')

# df = pd.read_csv('resultOnTrainSet.csv')
# print(len(df))
# df_hard = df[df['similarity'] < 0.5]
# df_easy = df.drop(df_hard.index)
# print(len(df_hard))
# print(len(df_easy))
# w_h = len(df)/(len(df_hard)*2)
# w_e = len(df)/(len(df_easy)*2)
# df_hard['label'] = [0 for _ in range(len(df_hard))]
# df_hard['weight'] = [w_h for _ in range(len(df_hard))]
# df_easy['label'] = [1 for _ in range(len(df_easy))]
# df_easy['weight'] = [w_e for _ in range(len(df_easy))]
# df = pd.concat([df_hard, df_easy], ignore_index=True)
# print(len(df))
# print(df.columns)
# df.to_csv('bc.csv')

# X = [0.7136, 0.6992, 0.7144, 0.7147, 0.7137, 0.7142, 0.7146, 0.7102, 0.7056, 0.7130, 0.7149, 0.7048, 0.7141]
# y = [0.56205, 0.53756, 0.56264, 0.56266, 0.5604, 0.56127, 0.56266, 0.55478, 0.54748, 0.55915, 0.56281, 0.55472, 0.56119]
# plt.scatter(X, y)
#
# # 添加图表标题和坐标轴标签
# plt.title('CV&LB')
# plt.xlabel('CV')
# plt.ylabel('LB')
#
# # 显示图表
# plt.show()

# import numpy as np
#
# embeddings_clip = np.load('clip_embeddings.npy')
# embeddings_vit = np.load('vit_embeddings.npy')
# embeddings_vit_base = np.load('vit_base_ebs.npy')
# embeddings_ofa = np.load('ofa_embeddings.npy')
# embeddings_ci = np.load('ci_embeddings_new.npy')


def cal(arr):
    # arr = arr.reshape(-1, 384)


    # norms = np.linalg.norm(arr, axis=1)  # 计算每个向量的模
    # arr = arr / norms[:, np.newaxis]  # 将每个向量缩放为单位向量

    # print(arr.shape)
    norms = np.linalg.norm(arr, axis=1)
    # print(norms.shape)
    mean = np.mean(norms)
    variance = np.var(norms)
    std = np.std(norms)

    print("均值：", mean)
    print("方差：", variance)
    print('标准差', std)
    print('*' * 20)
#
#
import numpy as np
# cal(np.load('image_features.npy'))
# cal(embeddings_vit)
# cal(embeddings_vit_base)
# cal(embeddings_ofa)
# cal(embeddings_ci)
#
# df = pd.read_csv('resultOnNew.csv')
# df1 = pd.read_csv('resultOnNewByTrained.csv')
# # print(len(df))
# # print(len(df[df['similarity'] < 0.6]))
# # import numpy as np
# # from matplotlib import pyplot as plt
# # embeddings_vit = np.load('vit_embeddings.npy')
# # embeddings_vit = embeddings_vit.reshape(-1, 384)
# # norms = np.linalg.norm(embeddings_vit, axis=1)  # 计算每个向量的模
# plt.hist(df['similarity'], bins=1000, range=(0, 1), alpha=0.4, label='pre')
# plt.hist(df1['similarity'], bins=1000, range=(0, 1), alpha=0.4, label='now')
# plt.show()
import os
import time
# import matplotlib.pyplot as plt

# df1 = pd.read_csv('csOnNew.csv')
# df = pd.read_csv('csOnValid.csv')
# df1 = pd.read_csv('resultOnTrainSet.csv')
# print(df.columns)
# df['filepath'] = pd.read_csv('train.csv')['filepath']
# print(df['prompt'].equals(pd.read_csv('train.csv')['prompt']))
# print(len(df))
# plt.hist(df['similarity'], bins=100, range=(0, 1), alpha=1)
# plt.bar(range(len(df)), df['similarity'], width=1)
# plt.bar(range(len(df)), df1['similarity'], width=1)

# plt.hist(df2['similarity'], bins=100, range=(0, 1), alpha=0.6)
# for num in [0.3953, 0.2861, 0.3445, 0.3530, 0.3655, 0.3691, 0.3782]:
#     plt.axvline(num, color='r', linestyle='--')
# plt.axvline(df['similarity'].mean(), color='b', linestyle='--')
# plt.axvline(df['similarity'].mean(), color='r', linestyle='--')
# print(df['similarity'].mean())
# plt.show()
# print(df['similarity'].mean())
# print(df1['similarity'].mean())
# print(sum(df['similarity'])/len(df))
# print(len(df[df['similarity'] > 0.36])/len(df))
# print(len(df[df['similarity'] < 0.28])/len(df))

# df = df[df['similarity'] > 0.3]
# filepaths = df['filepath'].to_list()
# prompts = df['prompt'].to_list()
# print(len(df))
# for i in range(len(df)):
#     img = plt.imread(filepaths[i])
#     plt.imshow(img)
#     plt.show()
#     print(prompts[i])
#     time.sleep(8)
# df = df[['prompt', 'filepath']]
# print(df.columns)
# df.to_csv('350k.csv')






# df1 = pd.read_csv('prompts2.csv')
# df2 = pd.read_csv('prompts1.csv')[:792]
# print(len(df1))
# print(len(df2))
# df = pd.concat([df1, df2], ignore_index=True)[['prompt']]
# print(len(df))
# print(df.columns)
# df.to_csv('prompt1.csv')

# df = pd.read_csv('prompt1.csv')
# df1 = df[:5000]
# df2 = df[5000:]
# print(len(df1))
# print(len(df2))
# df1.to_csv('prompts1_a.csv')
# df1.to_csv('prompts1_b.csv')


# import numpy as np
# import matplotlib.pyplot as plt
#
# # 示例数据
# sims = pd.read_csv('diffusion.csv')['similarity'][:100000]
# trn_sims = pd.read_csv('resultOnTrainSet.csv')['similarity'][:100000]
#
# # 按照sims的大小排序索引
# sorted_idx = np.argsort(trn_sims)
#
# # 使用排序后的索引绘制直方图
# plt.bar(range(len(sims)), height=sims[sorted_idx], width=1.0, color='red', alpha=0.3)
# plt.bar(range(len(trn_sims)), height=trn_sims[sorted_idx], width=1.0, color='blue', alpha=0.3)
# plt.axhline(0.4, color='r', linestyle='--')
# plt.axhline(0.3, color='r', linestyle='--')
# plt.axhline(0.2, color='r', linestyle='--')
# plt.show()
ebs_in = np.load('image_features.npy')
ebs_out = np.load('text_features.npy')
bias = ebs_in - ebs_out
# cal(bias)
# cal(ebs_in)
# cal(ebs_out)

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np

# 定义聚类数量和距离阈值
K = 1
distance_threshold = 8

# 创建一个512维度数据的示例数据集
X = bias

# 使用K-means算法对数据进行聚类
kmeans = KMeans(n_clusters=K, random_state=0).fit(X)

# 获取每个数据点所属聚类的标签和聚类中心的坐标
# labels = kmeans.labels_
# centers = kmeans.cluster_centers_
#
# # 计算每个数据点到其所属聚类中心的距离
# distances = pairwise_distances_argmin_min(X, centers)[1]
# plt.hist(distances, bins=100)
# plt.show()
#
# # 获取距离聚类中心过远的样本的索引
# outlier_idx = np.where(distances > distance_threshold)[0]
# np.save('distance.npy', distances)
# # 输出距离过远的样本数量
# print('Number of outliers:', len(outlier_idx))




# import numpy as np
# import matplotlib.pyplot as plt
#
# # 定义余弦相似度函数
# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
#
# vectors = bias[:2000]
#
# # 创建一个n x n的矩阵，其中n是向量数量
# n = len(vectors)
# similarity_matrix = np.zeros((n, n))
#
# # 对于矩阵中的每一对向量，计算它们之间的余弦相似度，并将其放入矩阵中
# for i in range(n):
#     for j in range(i, n):
#         similarity_matrix[i, j] = cosine_similarity(vectors[i], vectors[j])
#         similarity_matrix[j, i] = similarity_matrix[i, j]
#
# # 使用Matplotlib的imshow函数绘制热力图
# plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.show()
