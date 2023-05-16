import torch
import clip
from PIL import Image


text = clip.tokenize(["a dying man", "a man is lying along aside a drive road", "a car is passing by a man"]).to(device)
#
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)

import pandas as pd
from nltk.tokenize import word_tokenize
import gc

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

batch_size = 0

df = pd.read_csv('train.csv')
# df = df[df['prompt'].map(lambda p: len(p.split()) <= 77)]

prompts = df['prompt']
image_paths = df['filepath']
prompts = clip.tokenize(prompts.to_list(), truncate=True).to(device)
prompts_features = model.encode_text(prompts)

import torch
from tqdm import tqdm
# 分批进行编码
batch_size = 1
prompts = df['prompt'].tolist()
n_batches = (len(prompts) + batch_size - 1) // batch_size

# features = []
for i in tqdm(range(n_batches)):
    torch.cuda.empty_cache()
    gc.collect()
    batch_prompts = prompts[i*batch_size : (i+1)*batch_size]
    batch_inputs = clip.tokenize(batch_prompts, truncate=True).to(device)
    batch_features = model.encode_text(batch_inputs)
    # features.append(batch_features)



# 合并批次的编码结果
features = torch.cat(features)



from torch.utils.data import Dataset, DataLoader

image = preprocess([Image.open("/root/autodl-tmp/kaggle/PromptPredict/images/0/10.png"),
                    Image.open("/root/autodl-tmp/kaggle/PromptPredict/images/0/11.png"),
                    Image.open("/root/autodl-tmp/kaggle/PromptPredict/images/0/12.png")]).unsqueeze(0).to(device)

class ImageDataset(Dataset):
    def __init__(self, paths):
        self.image_paths = paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = preprocess(Image.open(self.image_paths[item])).unsqueeze(0).to(device)
        return image


dataloader = DataLoader(dataset=ImageDataset(image_paths),
                        batch_size=batch_size, shuffle=True, num_workers=8)

with torch.no_grad():
    for images in dataloader:
        image_features = model.encode_image(images)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()