import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm import tqdm

import torch
import clip
import torch.nn.functional as F

import gc

# load data
df = pd.read_csv('valid.csv')
# df2 = pd.read_csv('pis1_b.csv')
# print(len(df1))
# print(len(df2))
# df = pd.concat([df1, df2], ignore_index=True)
print(len(df))
prompts = df['prompt'].to_list()
filepaths = df['filepath'].to_list()

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

sims = []
with torch.no_grad():
    for i in tqdm(range(len(prompts))):
        prompt = clip.tokenize(prompts[i], truncate=True).to(device)
        image = preprocess(Image.open(filepaths[i]).convert('RGB')).unsqueeze(0).to(device)
        image_eb = model.encode_image(image)
        text_eb = model.encode_text(prompt)
        sims.append(F.cosine_similarity(image_eb, text_eb).cpu().item())

df = pd.DataFrame({'prompt': prompts, 'similarity': sims})
df.to_csv('csOnValid.csv')

