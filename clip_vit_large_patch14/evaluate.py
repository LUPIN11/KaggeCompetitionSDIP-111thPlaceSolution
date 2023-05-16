import os
import sys
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers import AutoProcessor
from torch.utils.data import DataLoader

sys.path.append('/root/autodl-tmp/kaggle/img2text/input/sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
sentence_encoder = SentenceTransformer(
    "/root/autodl-tmp/kaggle/img2text/input/sentence-transformers-222/all-MiniLM-L6-v2")


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
model.load_state_dict(torch.load('./ckpt/1|0.7005.pth'))
model.to(device)


def get_data():
    # df = pd.read_csv('/root/autodl-tmp/kaggle/img2text/input/sd2.csv')
    df = pd.read_csv('train0.csv')
    print(len(df))
    filepath = df['filepath'].copy().to_list()
    images = df['filepath']
    prompts = df['prompt'].to_list()
    ebs = sentence_encoder.encode(df["prompt"].to_numpy(), batch_size=512, show_progress_bar=True, device="cuda")
    return images, ebs, prompts, filepath


images, ebs, prompts, filepath = get_data()


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


dataloader = DataLoader(dataset=Dataset(images, ebs), batch_size=32, shuffle=False, num_workers=6, drop_last=False)
similarities = []
# norms = []
model.eval()
with torch.no_grad():
    for images, ebs in tqdm(dataloader, leave=False):
        images, ebs = images.to(device), ebs.to(device)
        pred = model(images)
        # norms.extend(torch.norm(pred, dim=1).cpu().tolist())
        similarities.extend(F.cosine_similarity(pred, ebs, dim=1).tolist())

# import statistics
#
#
# mean = statistics.mean(norms)
# variance = statistics.variance(norms)
#
# print("均值：", mean)
# print("方差：", variance)


df = pd.DataFrame({'prompt': prompts, 'similarity': similarities, 'filepath': filepath})
# print(len(df))
df.to_csv('resultOnTrain0.csv')
plt.hist(similarities, bins=100, range=(0, 1))
plt.title('Similarities Distribution')
plt.show()
# df = df[df['similarity'] < 0.5].copy()
# df.to_csv('hard_sd2.csv')
# print(len(df))
# df2 = pd.read_csv('/root/autodl-tmp/kaggle/fliter/train1.csv')
# mask = df['prompt'].isin(df2['prompt'])
#
# # 根据布尔索引 mask 筛选符合条件的行，生成新的 DataFrame
# new_df = df.loc[mask][['prompt', 'filepath']]
# print(len(new_df))
# print(new_df.columns)
# df3 = pd.read_csv('train.csv')[['prompt', 'filepath']]
# df = pd.concat([new_df, df3], ignore_index=True)
# print(len(df))
# print(df.columns)
# df.to_csv('train2.csv')


