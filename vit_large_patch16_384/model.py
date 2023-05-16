from collections.abc import Iterable

import timm
import torch
from torch import nn

from config import *


def load_pretrained_model(model):
    for name, child in model.named_children():
        # if name in ['patch_embed', 'pos_drop', 'norm_pre']:
        if name in ['patch_embed', 'pos_drop', 'norm_pre', 'norm', 'fc_norm']:
            if isinstance(child, Iterable):
                for idx, layer in enumerate(child):
                    for param in layer.parameters():
                        param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = False
        if name == 'blocks':  # layer idx ranges in 0~23
            for idx, layer in enumerate(child):
                # if idx <= (23 - 12):
                for param in layer.parameters():
                    param.requires_grad = False
    return model


vit = timm.create_model(MODEL_NAME, pretrained=True)
# vit.set_grad_checkpointing() # 这个控制显存和速度之间的平衡了
vit = load_pretrained_model(vit)


class Model(nn.Module):
    def __init__(self, pre_trained_model):
        super(Model, self).__init__()
        self.vit = pre_trained_model
        self.fc = nn.Sequential(
            nn.Linear(1000, 900),
            nn.ReLU(),
            nn.Linear(900, 800),
            nn.ReLU(),
            nn.Linear(800, 700),
            nn.ReLU(),
            nn.Linear(700, 600),
            nn.ReLU(),
            nn.Linear(600, 500),
            nn.ReLU(),
            nn.Linear(500, 384),
        )

    def forward(self, x):
        out = self.vit(x)
        return self.fc(out)
        # return out


model = Model(vit)
if CONTINUE:
    model.load_state_dict(torch.load(MODEL_LOAD_PATH))
model.to(DEVICE)
