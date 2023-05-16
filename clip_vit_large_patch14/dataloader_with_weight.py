import sys
from PIL import Image
import pandas as pd
from transformers import AutoProcessor
from torch.utils.data import DataLoader

sys.path.append('/root/autodl-tmp/kaggle/img2text/input/sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer

from config import *

clip_processor = AutoProcessor.from_pretrained("./offline_model")


def get_train_test_split():
    sentence_encoder = SentenceTransformer(
        "/root/autodl-tmp/kaggle/img2text/input/sentence-transformers-222/all-MiniLM-L6-v2")
    df_train = pd.read_csv('weighted_train_s.csv')
    df_valid = pd.read_csv('valid.csv')
    train_ebs = sentence_encoder.encode(df_train["prompt"].to_numpy(), batch_size=512, show_progress_bar=True,
                                        device="cuda")
    valid_ebs = sentence_encoder.encode(df_valid["prompt"].to_numpy(), batch_size=512, show_progress_bar=True,
                                        device="cuda")
    train_images = df_train['filepath'].to_list()
    valid_images = df_valid['filepath'].to_list()
    train_weights = df_train['weight']
    valid_weights = [1 for _ in range(len(valid_images))]

    print(f"train set size: {len(train_images)}, valid set size: {len(valid_images)}")
    return train_images, train_ebs, valid_images, valid_ebs, train_weights, valid_weights


class Dataset:
    def __init__(self, image_paths, targets, weights, clip_processor=clip_processor):
        self.images = image_paths
        self.labels = targets
        self.weights = weights
        self.input_processor = clip_processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = Image.open(self.images[item])
        image = self.input_processor(images=image, return_tensors='pt').pixel_values.squeeze(0)
        target = self.labels[item]
        weight = self.weights[item]
        return image, target, weight


train_images, train_targets, valid_images, valid_targets, train_weights, valid_weights = get_train_test_split()
train_dataloader = DataLoader(dataset=Dataset(train_images, train_targets, train_weights),
                              batch_size=batch_size, shuffle=True, num_workers=8)
valid_dataloader = DataLoader(dataset=Dataset(valid_images, valid_targets, valid_weights),
                              batch_size=batch_size, shuffle=False, num_workers=8)
