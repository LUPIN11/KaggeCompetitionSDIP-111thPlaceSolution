import sys
from PIL import Image
import pandas as pd
from transformers import AutoProcessor
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


sys.path.append('/root/autodl-tmp/kaggle/img2text/input/sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer

from config import *

clip_processor = AutoProcessor.from_pretrained("./offline_model")


def get_train_test_split():
    sentence_encoder = SentenceTransformer(
        "/root/autodl-tmp/kaggle/img2text/input/sentence-transformers-222/all-MiniLM-L6-v2")
    df_train = pd.read_csv('train350k.csv')[['prompt', 'filepath']]
    # df_0 = pd.read_csv('pis0.csv')[['prompt', 'filepath']]
    # df_0_t, df_0_v = train_test_split(df_0, test_size=0.05, random_state=42)
    # df_train = pd.concat([df_train, df_0_t], ignore_index=True)
    # df_train = df_train[df_train['prompt'].map(lambda x: 5 <= len(x.split()) <= 77)]
    # df_valid = pd.read_csv('valid0.csv')[['prompt', 'filepath']]
    df_valid = pd.read_csv('valid.csv')[['prompt', 'filepath']]
    # df_valid = pd.concat([df_valid, df_0_v], ignore_index=True)
    # df_valid = df_valid[df_valid['prompt'].map(lambda x: 5 <= len(x.split()) <= 77)]
    train_ebs = sentence_encoder.encode(df_train["prompt"].to_numpy(), batch_size=512, show_progress_bar=True,
                                        device="cuda")
    valid_ebs = sentence_encoder.encode(df_valid["prompt"].to_numpy(), batch_size=512, show_progress_bar=True,
                                        device="cuda")
    train_images = df_train['filepath'].to_list()
    valid_images = df_valid['filepath'].to_list()
    print(f"train set size: {len(train_images)}, valid set size: {len(valid_images)}")
    return train_images, train_ebs, valid_images, valid_ebs


class Dataset:
    def __init__(self, image_paths, targets, clip_processor=clip_processor):
        self.images = image_paths
        self.labels = targets
        self.input_processor = clip_processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = Image.open(self.images[item])
        image = self.input_processor(images=image, return_tensors='pt').pixel_values.squeeze(0)
        target = self.labels[item]
        return image, target


train_images, train_targets, valid_images, valid_targets = get_train_test_split()
train_dataloader = DataLoader(dataset=Dataset(train_images, train_targets),
                              batch_size=batch_size, shuffle=True, num_workers=8)
valid_dataloader = DataLoader(dataset=Dataset(valid_images, valid_targets),
                              batch_size=batch_size, shuffle=False, num_workers=8)
