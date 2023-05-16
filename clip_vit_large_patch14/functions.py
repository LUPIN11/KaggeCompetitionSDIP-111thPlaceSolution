import os
import sys
import random
import unicodedata
import numpy as np
import torch
from torch import nn

ST_MODULE_PATH = '/root/autodl-tmp/kaggle/img2text/input/sentence-transformers-222'
sys.path.append(ST_MODULE_PATH)
from sentence_transformers import SentenceTransformer

st_model = SentenceTransformer('/root/autodl-tmp/kaggle/img2text/input/sentence-transformers-222/all-MiniLM-L6-v2',
                               device='cpu')


def cosine_similarity_loss(pred, target, weights=None):
    if weights is not None:
        cos = nn.CosineSimilarity(dim=1)
        output = -cos(pred, target)
        weighted_output = output * weights
        return weighted_output.mean()
    else:
        cos = nn.CosineSimilarity(dim=1)
        output = -cos(pred, target).mean()
        return output


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def load_vocab(txt_path):
    with open(txt_path, 'r') as file:
        words_list = [line.strip() for line in file]
        words_set = set(words_list)
    return words_list, words_set


def semantic_similarity(prompt1, prompt2):
    eb1 = st_model.encode(prompt1, show_progress_bar=False, convert_to_tensor=True).unsqueeze(0)
    eb2 = st_model.encode(prompt2, show_progress_bar=False, convert_to_tensor=True).unsqueeze(0)
    return torch.nn.functional.cosine_similarity(eb1, eb2)


def is_english_only(string):
    for s in string:
        cat = unicodedata.category(s)
        if not cat in ['Ll', 'Lu', 'Nd', 'Po', 'Pd', 'Zs']:
            return False
    return True

def remove_prefix(sentence):
    return sentence.replace("try imagine this imaginative picture: ", "")