import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

# config
EMBEDDING_LENGTH = 384
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCALE = 'base'
INPUT_SIZE = 224
MODEL_NAME = f'vit_{SCALE}_patch16_{INPUT_SIZE}'
SAVED_MODEL_PATH = '/kaggle/input/shalewo/vit_large_patch16_384_0_64_0.0001_0.6472.pth'


# dataset
class DiffusionTestDataset(Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = self.transform(image)
        return image


images = list(Path('/kaggle/input/stable-diffusion-image-to-prompts/images').glob('*.png'))
transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
dataset = DiffusionTestDataset(images, transform)
dataloader = DataLoader(
    dataset=dataset,
    shuffle=False,
    batch_size=BATCH_SIZE,
    pin_memory=True,
    num_workers=2,
    drop_last=False
)

# imgId_eId
imgIds = [i.stem for i in images]
imgId_eId = [
    '_'.join(map(str, i)) for i in zip(
        np.repeat(imgIds, EMBEDDING_LENGTH),
        np.tile(range(EMBEDDING_LENGTH), len(imgIds)))]

# model
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=384)
state_dict = torch.load(SAVED_MODEL_PATH)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()


preds = []
for X in tqdm(dataloader, leave=False):
    X = X.to(DEVICE)
    with torch.no_grad():
        X_out = model(X)
        preds.append(X_out.cpu().numpy())

prompt_embeddings = np.vstack(preds).flatten()
submission = pd.DataFrame(
    index=imgId_eId,
    data=prompt_embeddings,
    columns=['val']
).rename_axis('imgId_eId')
submission.to_csv('submission.csv')