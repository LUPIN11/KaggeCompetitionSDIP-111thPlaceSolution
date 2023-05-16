# import pandas as pd
#
# # 创建一个示例 DataFrame
# df = pd.read_csv('new_prompts_gpt.csv')
# # 定义一个函数，用于去掉每个句子开头的字符串
# def remove_prefix(sentence):
#     sentence = sentence.replace('\n', ' ')
#     return sentence.replace("try imagine this imaginative picture: ", "")
#
# # 使用 apply() 方法将函数应用到 DataFrame 的每一行，并存储为一个新的属性
# df['prompt'] = df['prompt'].apply(lambda x: ' '.join(remove_prefix(sentence) for sentence in x.split('. \n')))
#
# # 打印处理后的结果
# df.to_csv('new_prompts_gpt.csv')
import numpy as np
import pandas as pd
import torch

# 创建一个布尔数组，作为索引数组
# bool_array = np.array([True, False, True])
#
# # 创建一个三维数组
# nd_array = np.array([
#     [[1, 2, 3], [4, 5, 6]],
#     [[7, 8, 9], [10, 11, 12]],
#     [[13, 14, 15], [16, 17, 18]]
# ])
#
# # 使用布尔数组作为索引数组，选择第一个维度为 True 的元素
# selected = nd_array[bool_array]
#
# print(selected)

# s = 'a lone soldier sci-fi retro walking on a dead planet while in the horizon you can see a beutiful sunset digital painting, artstation, concept art, soft light, hdri, smooth, sharp focus, illustration, fantasy, intricate, elegant, highly detailed, D&D, matte painting, in the style of Greg Rutkowski and Alphonse Mucha and artemisia, 8 k, highly detailed, jurgens, rutkowski, bouguereau, pastoral, rustic, georgic'
# print(len(s.split()))

# import nltk
# from nltk.tokenize import word_tokenize
#
# # nltk.download('punkt')  # 下载必要的数据
#
# sentence = "I love to use natural language processing tools like NLTK, but sometimes it can be challenging!"
# tokens = word_tokenize(sentence)
#
# print(tokens)

#
# from functions import *
#
# encoder = SentenceTransformer("/root/autodl-tmp/kaggle/img2text/input/sentence-transformers-222/all-MiniLM-L6-v2")
# df = pd.read_csv('valid.csv')
#
# vectors = encoder.encode(df["prompt"].to_numpy(), batch_size=512, show_progress_bar=True, device="cuda",
#                         convert_to_tensor=True)
# norms = torch.norm(vectors, dim=1)
# for norm in norms:
#     print(norm.item())

# import pandas as pd

# df1 = pd.read_csv('prompts1.csv')
# df2 = pd.read_csv('prompts2.csv')
# print(len(df1))
# print(len(df2))
# df3 = pd.concat([df1, df2], ignore_index=True)
# df3 = df3[['prompt']]
# print(len(df3))
# print(df3.columns)
# df3.to_csv('prompts1.csv')

# df = pd.concat([df1, df2, df3], ignore_index=True)
# print(len(df))
# df.to_csv('p.csv')

# df = pd.read_csv('prompts1.csv')
# df = pd.read_csv('p.csv')
# df['prompt'] = df['prompt'].str.replace('try image this picture', '', regex=False)
# df.to_csv('p.csv')
#
# df = pd.read_csv('pis1_b.csv')
# print(len(df))

# from PIL import Image
# import torch
# import torch.nn.functional as F
# import clip
# from matplotlib import pyplot as plt
#
# from sd2 import sd2
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# model, preprocess = clip.load("ViT-B/32", device=device)
#
# prompt = '''ultrasaurus holding a black bean taco in the woods, near an identical cheneosaurus'''
# text = clip.tokenize(prompt).to(device)
#
# images = []
# for _ in range(100):
#     image = sd2.generate(prompt).convert('RGB')
#     image_input = preprocess(image).unsqueeze(0).to(device)
#     images.append(image_input)
#
# text_features = model.encode_text(text)
# sims = []
# image_ebs = []
# with torch.no_grad():
#     for image in images:
#         image_features = model.encode_image(image)
#         # cs = F.cosine_similarity(image_features, text_features)
#         image_ebs.extend(image_features.cpu().numpy())
#         # sims.append(cs.cpu().item())
#     image_ebs.extend(text_features.cpu().numpy())
# # print(sims)
# # plt.hist(sims, bins=120, range=(0.2, 0.5))
# # plt.axvline(0.3530, color='r', linestyle='--')
# # plt.show()
# image_ebs = np.array(image_ebs)
# print(image_ebs.shape)
# import numpy as np
# from scipy.spatial.distance import pdist, squareform
# import matplotlib.pyplot as plt
#
# # 生成k个随机向量
# k = 5
# vectors = image_ebs
#
# # 计算余弦相似度
# distances = pdist(vectors, metric='cosine')
# similarity_matrix = 1 - squareform(distances)
#
# # 绘制热力图
# plt.imshow(similarity_matrix, cmap='coolwarm')
# plt.colorbar()
# plt.show()
from functions import *
import numpy as np
# df = pd.read_csv('p.csv')
# print(len(df))
# X = st_model.encode(df["prompt"].to_numpy(), batch_size=128, show_progress_bar=True, device='cuda')
# print(X.shape)
# # np.save('ebs400k.npy', X)
# ebs = np.load('new_ebs.npy')
# print(ebs.shape)
# def normalize(arr):
#     norms = np.linalg.norm(arr, axis=1, keepdims=True)
#     return arr / norms
# 
# a = np.array([[1,1,1],
#               [10,4,5]])
# print(normalize(a))
# 
# cs = torch.nn.CosineSimilarity(dim=1)
# print(cs(torch.Tensor([[1,2],
#                        [3,4]]),
#          torch.Tensor([[1,2],
#                        [3,4]])))

import pickle

# 打开文件并加载对象
# with open("hardcoded_prompts_V0.pk", "rb") as file:
#     generated_prompts = pickle.load(file)
#
# print(type(generated_prompts))
# print(len(generated_prompts))
# df = pd.DataFrame({'prompt': generated_prompts})
# print(generated_prompts[1])
# df.to_csv('hardcodeprompts.csv')
from matplotlib import pyplot as plt
plt.plot([0.1, 0.15, 0.3], [0.5866, 0.58701, 0.58773])
plt.show()