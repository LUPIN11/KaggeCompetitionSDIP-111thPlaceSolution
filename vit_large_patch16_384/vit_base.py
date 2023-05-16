import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.preprocessing import normalize

class CFG:
    model_path = '/root/autodl-tmp/kaggle/img2text/ckpt/vit_base_patch16_224.pth'
    model_name = 'vit_base_patch16_224'
    input_size = 224
    batch_size = 64


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


def predict(
        images,
        model_path,
        model_name,
        input_size,
        batch_size
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(degrees=10),

        # transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = DiffusionTestDataset(images, transform)
    dataloader = DataLoader(
        dataset=dataset,
        shuffle=False,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2,
        drop_last=False
    )

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=384
    )
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tta_preds = None
    for _ in range(2):
        preds = []
        for X in tqdm(dataloader, leave=False):
            X = X.to(device)

            with torch.no_grad():
                X_out = model(X).cpu().numpy()
                # L2 normalize -- Start
                X_out = X_out / (np.abs(X_out).max(axis=-1,
                                                   keepdims=True) + 0.0000001)  # To avoid to overflow at normalize()
                X_out = normalize(X_out)
                # L2 normalize -- End
                preds.append(X_out)

        if tta_preds is None:
            tta_preds = np.vstack(preds).flatten()
        else:
            tta_preds += np.vstack(preds).flatten()

    return tta_preds / 2

# images = list(Path('/kaggle/input/stable-diffusion-image-to-prompts/images').glob('*.png'))
# imgIds = [i.stem for i in images]
# EMBEDDING_LENGTH = 384
# imgId_eId = [
#     '_'.join(map(str, i)) for i in zip(
#         np.repeat(imgIds, EMBEDDING_LENGTH),
#         np.tile(range(EMBEDDING_LENGTH), len(imgIds)))]
images = pd.read_csv('valid.csv')['filepath']


embeddings = predict(images, CFG.model_path, CFG.model_name, CFG.input_size, CFG.batch_size)

np.save('vit_base_ebs.npy', embeddings)
