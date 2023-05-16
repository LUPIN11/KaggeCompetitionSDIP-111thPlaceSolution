import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
import sys
import torch
from sklearn.model_selection import train_test_split
from glob import glob
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from pathlib import Path
from transformers import AutoModel, AutoProcessor
sys.path.append('/root/autodl-tmp/kaggle/img2text/input/sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer


clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
BATCHSIZE = 32
SAVE_OPT_CKP = True
SAVE_MODEL_CKP = True
UNFREEZE_START = 18  # set it to lower number when significantly more samples are included.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

run_name = f'clip224-l18'


def cosine_similarity_loss(pred, target):
    cos = nn.CosineSimilarity(dim=1)
    output = -cos(pred, target).mean()
    return output


def get_train_test_split():
    sentence_encoder = SentenceTransformer(
        "/root/autodl-tmp/kaggle/img2text/input/sentence-transformers-222/all-MiniLM-L6-v2")
    df_train = pd.read_csv('train.csv').iloc[:10000, :]
    print(len(df_train))
    df_valid = pd.read_csv('valid.csv')
    train_ebs = sentence_encoder.encode(df_train["prompt"].to_numpy(), batch_size=512, show_progress_bar=True, device="cuda")
    valid_ebs = sentence_encoder.encode(df_valid["prompt"].to_numpy(), batch_size=512, show_progress_bar=True, device="cuda")
    train_images = df_train['filepath'].to_list()
    valid_images = df_valid['filepath'].to_list()
    return train_images, train_ebs, valid_images, valid_ebs


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        clip = AutoModel.from_pretrained("openai/clip-vit-large-patch14")
        self.vision = clip.vision_model
        self.fc = nn.Linear(1024, 384)

    def forward(self, x):
        out = self.vision(x)['pooler_output']
        return self.fc(out)


def load_pretrained_model():
    model = Net()

    trainable_model_weights = False
    for name, child in model.named_children():
        if name == 'vision':
            for pn, p in child.named_parameters():
                if str(UNFREEZE_START) in pn:
                    """start unfreezing layer , the weights are trainable"""
                    trainable_model_weights = True
                p.requires_grad = trainable_model_weights
                if p.requires_grad:
                    print(f"{pn} is set to be trainable.")

    return model.to(device)


class IMGDataset:
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


if __name__ == "__main__":
    """main training"""
    Path(f"../{run_name}").mkdir(exist_ok=True)

    NEPOCH = 25
    BestEpoch = 0
    BestSim = 0
    train_images, train_targets, test_images, test_targets = get_train_test_split()

    print(f"test size: {len(test_images)}, train size: {len(train_images)}")

    nn_model = load_pretrained_model()
    # nn_model = torch.compile(nn_model)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, nn_model.parameters()), lr=1e-4) # fused=True
    optimizer.zero_grad()
    test_dataloader = DataLoader(dataset=IMGDataset(test_images, test_targets),
                                 batch_size=BATCHSIZE, shuffle=False, num_workers=4)
    train_dataloader = DataLoader(dataset=IMGDataset(train_images, train_targets),
                                  batch_size=BATCHSIZE, shuffle=True, num_workers=4)

    for epoch in range(NEPOCH):
        epoch_loss = 0
        for s, batch_data in enumerate(tqdm(train_dataloader)):
            batch_images, batch_targets = batch_data
            batch_images, batch_targets = batch_images.to(device), batch_targets.to(device)
            pred = nn_model(batch_images)
            cosine_loss = cosine_similarity_loss(pred, batch_targets)
            loss = cosine_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += -cosine_loss.item()
        epoch_loss /= len(train_dataloader)
        print(f"epoch: {epoch}, training loss: {epoch_loss}")

        """test loss"""
        epoch_loss = 0
        with torch.no_grad():
            for batch_images, batch_targets in tqdm(test_dataloader):
                batch_images, batch_targets = batch_images.to(device), batch_targets.to(device)
                pred = nn_model(batch_images)
                loss = -cosine_similarity_loss(pred, batch_targets)
                epoch_loss += loss.item()
            epoch_loss /= len(test_dataloader)
        print(f"epoch: {epoch}, test loss: {epoch_loss}")

        if epoch_loss > BestSim:
            BestSim = epoch_loss
            BestEpoch = epoch + 1
            print(f"save best model at {BestSim} with epoch {BestEpoch}")
            if SAVE_MODEL_CKP:
                torch.save(nn_model.state_dict(), f"{run_name}.pt")
            if SAVE_OPT_CKP:
                torch.save(optimizer.state_dict(), f"{run_name}_opt.pt")

        if epoch - 3 > BestEpoch:
            print(f"early stop at {epoch + 1} with best epoch {BestEpoch} and test similarity {BestSim}.")
            break