import os
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import clip
import torch
from timm.utils import AverageMeter
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from functions import st_model
from sklearn.model_selection import train_test_split

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
lr = 1e-4
num_epochs = 100

clip_model, preprocess = clip.load("ViT-B/32", device='cpu')



def get_data():
    # embeddings(output)
    # prompts = pd.read_csv('/root/autodl-tmp/kaggle/img2text/prompts150k.csv')['prompt'].to_numpy()
    # ebs = st_model.encode(prompts, batch_size=512, show_progress_bar=True, device=device)
    # np.save('ebs150k.npy', ebs)
    ebs = np.concatenate((np.load('ebs1m.npy'), np.load('ebs400k.npy')), axis=0)
    # text features(input)
    features = np.concatenate((np.load('text_features1m.npy'), np.load('text_features400k.npy')), axis=0)
    # # valid
    # df = pd.read_csv('valid.csv')
    # filepaths = df['filepath'].to_list()
    # prompts = df['prompt'].to_numpy()
    # val_ebs = st_model.encode(prompts, batch_size=512, show_progress_bar=True, device=device)
    print('train set size: ', len(ebs))
    return ebs, features


class TextDataset(Dataset):
    def __init__(self, ebs, features):
        self.ebs = ebs
        self.features = features

    def __len__(self): return len(self.ebs)

    def __getitem__(self, item):
        return self.features[item], self.ebs[item]


ebs, features = get_data()
ebs_train, ebs_valid, features_train, features_valid = train_test_split(ebs, features, test_size=0.1, random_state=42)
print('train set size: ', ebs_train.shape)
print('valid set size: ', ebs_valid.shape)

train_dataloader = DataLoader(dataset=TextDataset(ebs_train, features_train),
                              batch_size=batch_size, shuffle=True, num_workers=6)
valid_dataloader = DataLoader(dataset=TextDataset(ebs_valid, features_valid),
                              batch_size=batch_size, shuffle=False, num_workers=6)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            torch.nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            torch.nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            torch.nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 384),
        )

    def forward(self, x):
        return self.layers(x)


model = Model()
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=lr)
optimizer.zero_grad()

# criterion = nn.CosineSimilarity(dim=1).to(device)
# criterion = nn.MSELoss().to(device)
cosine_loss = nn.CosineEmbeddingLoss(margin=0.95).to(device)
cosine_sim = torch.nn.CosineSimilarity(dim=1).to(device)

bst_cos = -1

for epoch in range(num_epochs):
    train_meters = {'cos': AverageMeter()}
    valid_meters = {'cos': AverageMeter()}
    model.train()
    bar = tqdm(train_dataloader, leave=False)
    for X, y in bar:
        X, y = X.to(device), y.to(device)
        output = model(X)
        # loss = criterion(output, y)
        target = torch.ones(X.size(0)).to(device)
        loss = cosine_loss(output, y, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_meters['cos'].update(cosine_sim(output, y).mean().item(), n=X.shape[0])
        bar.set_postfix(trn_cos=f'{train_meters["cos"].avg:.4f}')
    print(f"Epoch {epoch + 1:d} / train_cos={train_meters['cos'].avg:.4f}")
    model.eval()
    with torch.no_grad():
        bar = tqdm(valid_dataloader, leave=False)
        for X, y in bar:
            X, y = X.to(device), y.to(device)
            output = model(X)
            valid_meters['cos'].update(cosine_sim(output, y).mean().item(), n=X.shape[0])
            bar.set_postfix(val_cos=f'{valid_meters["cos"].avg:.4f}')
    print(f"Epoch {epoch + 1:d} / valid_cos={valid_meters['cos'].avg:.4f}")
    if valid_meters['cos'].avg > bst_cos:
        bst_cos = valid_meters['cos'].avg
        torch.save(model.state_dict(), f'./ckpt/512to384.pth')
        print('current model saved')

# class ImageDataset(Dataset):
#     def __init__(self, filepaths, val_ebs):
#         self.filepaths = filepaths
#         self.ebs = val_ebs
#
#     def __len__(self): return len(val_ebs)
#
#     def __getitem__(self, item):
#         image = Image.open(self.filepaths[item]).convert('RGB')
#         image = preprocess(image).unsqueeze(0)
#         image = clip_model.encode_image(image)
#         eb = self.ebs[item]
#         return image, eb
