# import pandas as pd
# import clip
# import torch
# from PIL import Image
# from tqdm import tqdm
# import torch.nn.functional as F
# from functions import st_model
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# df = pd.read_csv('train.csv')
# print(len(df))
# prompts = df['prompt'].to_list()
# filepaths = df['filepath'].to_list()
#
# model, preprocess = clip.load("ViT-L/14", device=device)
# # print(clip.available_models())
#
# text_features = []
#
# with torch.no_grad():
#     for i in tqdm(range(len(prompts))):
#         prompt = clip.tokenize(prompts[i], truncate=True).to(device)
#         text_eb = model.encode_text(prompt)[0]
#         text_features.append(text_eb.cpu().numpy().astype(np.float32))
#
# text_features = np.array(text_features)
# print(text_features.shape)
# np.save('text_features0.npy', text_features)
#
#
# image_features = []
#
#
# with torch.no_grad():
#     for i in tqdm(range(len(filepaths))):
#         image = Image.open(filepaths[i]).convert('RGB')
#         image = preprocess(image).unsqueeze(0).to(device)
#         image_ebs = model.encode_image(image)[0]
#         image_features.append(image_ebs.cpu().numpy().astype(np.float32))
#
#
# image_features = np.array(image_features)
# print(image_features.shape)
# np.save('image_features.npy', image_features)

import pandas as pd
from PIL import Image
from tqdm import tqdm

import clip
import torch
import torch.nn.functional as F

from matplotlib import pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(clip.available_models())
model, preprocess = clip.load("ViT-B/16", device=device)

df = pd.read_csv('train0.csv')[['prompt', 'filepath']]  # replace it with ur train set
prompts = df['prompt'].to_list()
filepaths = df['filepath'].to_list()
print(len(prompts))
sims = []
with torch.no_grad():
    for i in tqdm(range(len(df))):
        text_input = clip.tokenize(prompts[i], truncate=True).to(device)
        text_eb = model.encode_text(text_input)
        image = Image.open(filepaths[i]).convert('RGB')
        image = preprocess(image).unsqueeze(0).to(device)
        image_eb = model.encode_image(image)
        sims.append(F.cosine_similarity(text_eb, image_eb).cpu().item())

plt.hist(sims, bins=500, range=(0, 1))
plt.show()

sims = map(lambda x: x**5, sims)  # higher sim, higher weight
df['weight'] = pd.Series(sims)

df.to_csv('train0_weighted.csv')
