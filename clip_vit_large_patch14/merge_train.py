import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from models import Net
from timm.utils import AverageMeter
from functions import st_model
import torch.optim as optim
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-5
num_epochs = 50
batch_size = 32

# input
train_image_features = np.load('image_features.npy')
train_prompts = pd.read_csv('train.csv')['prompt']
train_prompt_embeddings = st_model.encode(train_prompts.to_numpy(), batch_size=512, show_progress_bar=True,
                                          device="cuda")

valid_image_features = np.load('valid_image_features.npy')
valid_prompts = pd.read_csv('valid.csv')['prompt']
valid_prompt_embeddings = st_model.encode(valid_prompts.to_numpy(), batch_size=512, show_progress_bar=True,
                                          device="cuda")


class MyDataset(Dataset):
    def __init__(self, features, ebs):
        self.features = features
        self.ebs = ebs

    def __len__(self):
        return len(self.ebs)

    def __getitem__(self, item):
        return self.features[item], self.ebs[item]


train_dataloader = DataLoader(MyDataset(train_image_features, train_prompt_embeddings),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)

valid_dataloader = DataLoader(MyDataset(valid_image_features, valid_prompt_embeddings),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)

# model
model = Net('./ckpt/512to512.pth', './ckpt/512to384.pth', 1)
# model = Net()
model.to('cuda')

# others
param_group = [
    {'params': model.clip_model.parameters(), 'lr': 1e-7},
    {'params': model.trans_model.parameters(), 'lr': 1e-7},
    {'params': model.fc1.parameters(), 'lr': 1e-4},
    {'params': model.fc2.parameters(), 'lr': 1e-4}
]

optimizer = optim.AdamW(param_group)
optimizer.zero_grad()

criterion = torch.nn.CosineSimilarity(dim=1).to('cuda')

# main
bst_cos = -1

for epoch in range(num_epochs):
    train_meters = {'cos': AverageMeter()}
    valid_meters = {'cos': AverageMeter()}
    model.train()
    bar = tqdm(train_dataloader, leave=False)
    for X, y in bar:
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = -criterion(output, y).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_meters['cos'].update(-loss.item(), n=X.shape[0])
        bar.set_postfix(trn_cos=f'{train_meters["cos"].avg:.4f}')
    print(f"Epoch {epoch + 1:d} / train_cos={train_meters['cos'].avg:.4f}")
    model.eval()
    bar = tqdm(valid_dataloader, leave=False)
    for X, y in bar:
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = criterion(output, y).mean()
        valid_meters['cos'].update(loss.item(), n=X.shape[0])
        bar.set_postfix(val_cos=f'{valid_meters["cos"].avg:.4f}')
    print(f"Epoch {epoch + 1:d} / valid_cos={valid_meters['cos'].avg:.4f}")
    if valid_meters['cos'].avg > bst_cos:
        bst_cos = valid_meters['cos'].avg
        torch.save(model.state_dict(), f'./ckpt/supermodel.pth')
        print('current model saved')
