import timm
import torch
import pandas as pd
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
from functions import *
from timm.utils import AverageMeter
from torchvision import transforms
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'vit_large_patch16_384'
MODEL_CKPT = './ckpt/vit_large_patch16_384_0_64_0.0001_0.6472.pth'
# load model
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=384)
model.load_state_dict(torch.load(MODEL_CKPT))
model.to('cuda')


# load data
class MyDataset(Dataset):
    def __init__(self, df):
        self.transform = transforms.Compose([
            transforms.Resize(384),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.df = df

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['filepath'])
        image = self.transform(image)
        prompt = row['prompt']
        return image, prompt, row['filepath']


class MyCollator:
    def __init__(self):
        self.st_model = SentenceTransformer('./input/sentence-transformers-222/all-MiniLM-L6-v2', device='cpu')

    def __call__(self, batch):
        images, prompts, filepaths = zip(*batch)
        images = torch.stack(images)
        prompt_embeddings = self.st_model.encode(prompts, show_progress_bar=False, convert_to_tensor=True)
        return images, prompt_embeddings, prompts, filepaths


collator = MyCollator()
df = pd.read_csv('valid.csv')
print(len(df))
dataset = MyDataset(df)
dataloader = DataLoader(
    dataset=dataset,
    shuffle=True,
    batch_size=16,
    pin_memory=True,
    num_workers=8,
    drop_last=True,
    collate_fn=collator
)
# criterion
criterion = nn.CosineEmbeddingLoss()
# eval
# valid
similarities = []
prompts = []
filepaths = []
model.eval()
for X, y, prompt, filepath in tqdm(dataloader, leave=False, ncols=100):
    X, y = X.to('cuda'), y.to('cuda')
    with torch.no_grad():
        X_out = model(X)
    cosine_similarities = F.cosine_similarity(X_out, y, dim=1).tolist()
    similarities.extend(cosine_similarities)
    prompts.extend(prompt)
    filepaths.extend(filepath)
df_r = pd.DataFrame({'prompt': prompts, 'similarity': similarities, 'filepath': filepaths})
# df_r = df_r.merge(df, on='prompt')
# print(df_r.columns)
df_r.to_csv('vitOnValidSet.csv')
print(df_r['similarity'].mean())
plt.hist(similarities, bins=1000, range=(0, 1))
plt.title('Similarities Distribution')
plt.show()

