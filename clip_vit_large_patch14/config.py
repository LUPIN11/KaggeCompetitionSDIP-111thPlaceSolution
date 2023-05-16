import torch
import os
from functions import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unfreeze_start = 23  # encoder模块共23层
print(f'unfreeze_start: {unfreeze_start}')
batch_size = 64
num_epochs = 5
lr = 1e-4
seed_everything(42)
