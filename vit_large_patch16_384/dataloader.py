from PIL import Image

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sentence_transformers import SentenceTransformer

from config import *


class MyDataset(Dataset):
    def __init__(self, df, transform):
        st_model = SentenceTransformer(ST_MODEL_PATH, device='cpu')
        self.images = df['filepath']
        self.prompts = df['prompt']
        self.prompts = st_model.encode(self.prompts.to_numpy(), batch_size=512, show_progress_bar=True,
                                            device="cuda")
        self.transform = transform

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = self.transform(image)
        prompt = self.prompts[idx]
        return image, prompt


def dataloaders(train_df, vaild_df):
    transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    trn_dataset = MyDataset(train_df, transform)
    val_dataset = MyDataset(vaild_df, transform)
    train_dataloader = DataLoader(
        dataset=trn_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=6,
        drop_last=False,
    )
    valid_dataloader = DataLoader(
        dataset=val_dataset,
        shuffle=False,
        batch_size=BATCH_SIZE,
        num_workers=6,
        drop_last=False,
    )
    return train_dataloader, valid_dataloader


train_df = pd.read_csv(TRAIN_CSV_PATH)
valid_df = pd.read_csv(VALID_CSV_PATH)
train_dataloader, valid_dataloader = dataloaders(train_df, valid_df)
