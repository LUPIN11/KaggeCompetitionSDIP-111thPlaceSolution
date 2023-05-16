import torch
import timm
from transformers import AutoModel, AutoProcessor
from collections.abc import Iterable

# #
# model = timm.create_model('vit_base_patch16_224', pretrained=False)
clip = AutoModel.from_pretrained("openai/clip-vit-large-patch14")
# def load_pretrained_model(model):
#     trainable = False
#     for name, child in model.named_children():
#         print(name)
#         if name == 'vision':
#             for pn, p in child.named_parameters():
#                 if str(18) in pn:
#                     trainable = True
#                 p.requires_grad = trainable
#                 if p.requires_grad:
#                     print(f"{pn} is set to be trainable.")
#     return model
# load_pretrained_model(clip)
num_params = sum(p.numel() for p in clip.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")
# clip = AutoModel.from_pretrained("openai/clip-vit-large-patch14")

# def load_pretrained_model():
#     model = clip
#
#     trainable_model_weights = False
#     for name, child in model.named_children():
#         print(name)
#         if name == 'vision':
#             print('ok')
#         if isinstance(child, Iterable):
#             for idx, layer in enumerate(child):
#                 print(idx)
#         else:
#             pass
#     return model.to('cuda')
#
# load_pretrained_model()
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# similarities = np.random.random(10000)
# plt.hist(similarities, bins=100, range=(0, 1))
# plt.title('Similarities Distribution')
# plt.show()

# prompts = [
#         'hyper realistic photo of very friendly and dystopian crater',
#         'ramen carved out of fractal rose ebony, in the style of hudson river school',
#         'ultrasaurus holding a black bean taco in the woods, near an identical cheneosaurus',
#         'a thundering retro robot crane inks on parchment with a droopy french bulldog',
#         'portrait painting of a shimmering greek hero, next to a loud frill-necked lizard',
#         'an astronaut standing on a engaging white rose, in the midst of by ivory cherry blossoms',
#         'Kaggle employee Phil at a donut shop ordering all the best donuts, with a speech bubble that proclaims '
#         '"Donuts. It"s what“s for dinner!" '
#     ]
# print([len(prompt.split()) for prompt in prompts])
# print(sum([len(prompt.split()) for prompt in prompts])/len(prompts))

import pandas as pd

# df = pd.read_csv('6480on_train_1.csv')
# df = df[['prompt', 'similarity']].copy()
# print(df.columns)
# print(len(df[df['similarity'] < 0.3]))
# df_sorted = df.sort_values(by='similarity', ascending=True)
# df_min_similarity = df_sorted.head(10000)
# df1 = df_min_similarity.iloc[:5000, :]
# df2 = df_min_similarity.iloc[5000:, :]
# print(df1.columns)
# print(df2.columns)
# print(len(df1))
# print(len(df2))
# df1.to_csv('part3.csv')
# df1.to_csv('part4.csv')

#
# df1 = pd.read_csv('new1.csv')
# df2 = pd.read_csv('new2.csv')
# df3 = pd.read_csv('diffusiondb_121_train.csv')
# df4 = pd.read_csv('hard12n.csv')
# df = pd.concat([df1, df2, df3, df4], ignore_index=True)
# print(len(df))
# print(len(df1))
# print(len(df2))
# print(len(df3))
# print(len(df4))
# df.to_csv('train_1.csv')

# df4 = pd.read_csv('6472_on_diffusiondb_12n.csv')
# # print(len(df4[''].unique()))
# # print(len())
# df5 = pd.read_csv('diffusiondb_12n.csv')
# # print(len(df5[df5['prompt'].isin(df4[df4['similarity'] < 0.5]['prompt'].unique())]))
# df6 = df5[df5['prompt'].isin(df4[df4['similarity'] < 0.5]['prompt'].unique())]
# df6 = df6[['filepath', 'prompt']].copy()
# print(df6.columns)
# print(len(df6))
# print(len(df6['prompt'].unique()))
# df6.to_csv('hard12n.csv')

# import matplotlib.pyplot as plt
# df = pd.read_csv('6472_on_diffusiondb_121_train.csv')
# df = df[df['similarity'] < 0.3]
# lens = df['prompt'].map(lambda x: len(x.split()))
# plt.hist(lens, bins=100, range=(0, 100))
# plt.show()
#
# df3 = pd.read_csv('new3.csv')
# df3 = df3.iloc[:5200, :]
# print(len(df3))
# df4 = pd.read_csv('new4.csv')
# print(len(df4))
# df5 = pd.read_csv('train_1.csv')
# df5 = df5[['prompt', 'filepath']].copy()
# print(len(df5))
# df = pd.concat([df3, df4, df5], ignore_index=True)
# print(df.columns)
# print(len(df))
#
# df.to_csv('train_2.csv')

# df = pd.read_csv('valid.csv')
# df = df[['prompt', 'filepath']].copy()
# print(len(df))
# df = df.apply(lambda x: x.astype(str).apply(lambda y: '/root/autodl-tmp/' + y if 'filepath' in y else y))
# print(len(df))
# df.to_csv('valid.csv')
#
# import re
# text = "3 d, 4 k , 1 9 2 0 s, 3 girls, 2 3 year old, 2 3 4 apples, 35 mm , 565 mm , f 22 , of 33"
# text = re.sub(r'(?<=\d)\s(?=\d)', '', text)
# text = re.sub(r'\b(\d+)\s*([kKdDsS])\b', r'\1\2', text)
# text = re.sub(r'\b(\d+)\s*([yY]ear)\b', r'\1 \2', text)
# text = re.sub(r'\b(\d+)\s*mm\b', r'\1mm', text)
# text = re.sub(r'\bf\s*(\d+)', r'f\1', text)
# print(text)

import re

# # 定义需要处理的字符串
# s = 'abc123def45   6ghi789jkl,  1 9   30s, 89years, 1   man, 4  k ,   8k, 3d,   22mm, f12  ,   love22'
#
# result = re.sub(r'(?<=\S)(?=\d)|(?<=\d)(?=\S)', ' ', s)
# result = re.sub(r'\s{2,}', ' ', result)
# print(result)