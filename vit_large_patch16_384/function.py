import os
import sys
import random

import torch
import numpy as np
from scipy import spatial

ST_MODULE_PATH = '/root/autodl-tmp/kaggle/img2text/input/sentence-transformers-222'
sys.path.append(ST_MODULE_PATH)
from sentence_transformers import SentenceTransformer

st_model = SentenceTransformer('/root/autodl-tmp/kaggle/img2text/input/sentence-transformers-222/all-MiniLM-L6-v2',
                               device='cpu')


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def cosine_similarity(y_trues, y_preds):
    return np.mean([
        1 - spatial.distance.cosine(y_true, y_pred)
        for y_true, y_pred in zip(y_trues, y_preds)
    ])


def semantic_similarity(prompt1, prompt2):
    eb1 = st_model.encode(prompt1, show_progress_bar=False, convert_to_tensor=True).unsqueeze(0)
    eb2 = st_model.encode(prompt2, show_progress_bar=False, convert_to_tensor=True).unsqueeze(0)
    return torch.nn.functional.cosine_similarity(eb1, eb2)


if __name__ == "__main__":
    # sen1 = 'i am a robot, 4 k, red - white , you & me, 1 2 0 cm, @ @ rick @ @, love ! ! ! ! !'
    # sen2 = 'i am a robot, 4 k, red-white , you&me, 1 2 0 cm, @@rick@@, love!!!!!'
    # sen1 = 'a 33m view watercolor ink painting of broly as the god of war in the style of jean giraud in the style of moebius trending on artstation deviantart pinterest detailed realistic hd 8 k high resolution'
    # sen2 = 'a 33 m view watercolor ink painting of broly as the god of war in the style of jean giraud in the style of moebius trending on artstation deviantart pinterest detailed realistic hd 8 k high resolution'
    sen1 = 'who'
    sen2 = 'he'
    print(semantic_similarity(sen1, sen2))

    """"
    空格太多无所谓，有所谓是有无
    ab, 和 ab ，
    ab. 和 ab .
    都是没有区别的 
    @@evangelion@@与@ @ evangelion @ @没区别
    red-white和red - white没区别
    3 / 4和 3/4没区别
    """
