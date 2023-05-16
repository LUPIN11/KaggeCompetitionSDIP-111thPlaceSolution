import numpy as np
import pandas as pd
import torch
from functions import st_model
from models import Net


def normalize(arr):
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / norms


# input
valid_image_features = np.load('valid_image_features.npy')
# valid_image_features = normalize(valid_image_features)
valid_image_features = torch.from_numpy(valid_image_features).to('cuda')
# label
valid_prompts = pd.read_csv('valid.csv')['prompt']
valid_prompt_embeddings = st_model.encode(valid_prompts.to_numpy(), batch_size=512, show_progress_bar=True, device="cuda")
# valid_prompt_embeddings = normalize(valid_prompt_embeddings)
valid_prompt_embeddings = torch.from_numpy(valid_prompt_embeddings).to('cuda')


# model
model = Net('./ckpt/512to512.pth', './ckpt/512to384.pth', 1)  # 最佳是1.2
# model.load_state_dict(torch.load('./ckpt/supermodel.pth'))
model.to('cuda')
# criterion
criterion = torch.nn.CosineSimilarity(dim=1).to('cuda')
# eval
model.eval()
with torch.no_grad():
    output = model(valid_image_features)
# np.save('new_ebs.npy', output.cpu().numpy())
cos_sim = criterion(output, valid_prompt_embeddings).mean()
print('mean cosine simlarity on valid set: ', cos_sim.item())
