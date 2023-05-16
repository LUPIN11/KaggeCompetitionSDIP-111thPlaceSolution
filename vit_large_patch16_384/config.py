import sys
import warnings

import torch

from functions import seed_everything


SEED = 42
seed_everything(SEED)
# device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# filter warnings
warnings.filterwarnings('ignore')
# tqdm
BAR_LENGTH = 100
# pretrained model
CONTINUE = False
SCALE = 'large'  # base/large
INPUT_SIZE = 384  # 224/384
MODEL_NAME = f'vit_{SCALE}_patch16_{INPUT_SIZE}'
# hyperparameters
NUM_EPOCHS = 3
BATCH_SIZE = 16  # 64
LR = 1e-5  # 1e-4
# lr/batch_size越大泛化性能越好
MIN_LR = 1e-6
# filepath
ST_MODULE_PATH = './input/sentence-transformers-222/sentence-transformers'
sys.path.append(ST_MODULE_PATH)
ST_MODEL_PATH = './input/sentence-transformers-222/all-MiniLM-L6-v2'
TRAIN_CSV_PATH = 'train.csv'
VALID_CSV_PATH = 'valid.csv'
MODEL_SAVE_PATH = './ckpt'
MODEL_LOAD_PATH = './ckpt/vit_large_patch16_384|0|64|0.0001|0.6865.pth'

"""
train-cos/valid-cos/lb-cos after 1 epoch
(just head unfrozed)
0.5907-0.6865-0.52815(4 layers unfrozed)
0.5918-0.6894-0.52789(12 layers unfrozed)
0.5827-0.6842-0.52034(all block unfrozed)
"""