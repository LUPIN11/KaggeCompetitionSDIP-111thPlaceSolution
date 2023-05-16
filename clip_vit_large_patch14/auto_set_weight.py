import pandas as pd
import clip
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv('train.csv')[:10]
prompts = df['prompt'].to_list()
filepaths = df['filepath'].to_list()

# print(clip.available_models())
model, preprocess = clip.load("ViT-B/16", device=device)


sims = []
with torch.no_grad():
    for i in tqdm(range(len(df))):
        text_input = clip.tokenize(prompts[i], truncate=True).to(device)
        text_eb = model.encode_text(text_input)[0]
        image = Image.open(filepaths[i]).convert('RGB')
        image = preprocess(image).unsqueeze(0).to(device)
        image_eb = model.encode_image(image)[0]
        sims.append(F.cosine_similarity(text_eb, image_eb).cpu().item())
