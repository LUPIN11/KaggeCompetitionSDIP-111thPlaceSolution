from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import AutoModel
from transformers import AutoProcessor
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")


def cosine_similarity_loss(pred, target):
    cos = nn.CosineSimilarity(dim=1)
    output = -cos(pred, target).mean()
    return output


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        clip = AutoModel.from_pretrained("openai/clip-vit-large-patch14")
        self.vision = clip.vision_model
        self.fc = nn.Linear(1024, 384)

    def forward(self, x):
        out = self.vision(x)['pooler_output']
        return self.fc(out)


model = Model()
model.load_state_dict(torch.load('./ckpt/0_0.7093.pth'))
model.to(device)


def get_data():
    images = list(Path('/kaggle/input/stable-diffusion-image-to-prompts/images').glob('*.png'))
    imgIds = [i.stem for i in images]
    imgId_eId = [
        '_'.join(map(str, i)) for i in zip(
            np.repeat(imgIds, 384),
            np.tile(range(384), len(imgIds)))]
    return images, imgId_eId


images, imgId_eId = get_data()


class Dataset:
    def __init__(self, image_paths, clip_processor=clip_processor):
        self.images = image_paths
        self.input_processor = clip_processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = Image.open(self.images[item])
        image = self.input_processor(images=image, return_tensors='pt').pixel_values.squeeze(0)
        return image


dataloader = DataLoader(dataset=Dataset(images),batch_size=32, shuffle=False, num_workers=4, drop_last=False)
preds = []
model.eval()
with torch.no_grad():
    for images in dataloader:
        images = images.to(device)
        pred = model(images)
        preds.append(pred.cpu().numpy())

prompt_embeddings = np.vstack(preds).flatten()
submission = pd.DataFrame(
    index=imgId_eId,
    data=prompt_embeddings,
    columns=['val']
).rename_axis('imgId_eId')
submission.to_csv('submission.csv')



