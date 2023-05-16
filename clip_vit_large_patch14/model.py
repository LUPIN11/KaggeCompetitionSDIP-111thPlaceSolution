from torch import nn
from transformers import AutoModel

from config import *


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        clip = AutoModel.from_pretrained("./offline_model")
        self.vision = clip.vision_model
        self.fc = nn.Linear(1024, 384)

    def forward(self, x):
        out = self.vision(x)['pooler_output']
        return self.fc(out)


def load_pretrained_model(model):
    trainable = False
    for name, child in model.named_children():
        if name == 'vision':
            for pn, p in child.named_parameters():
                if str(unfreeze_start) in pn:
                    trainable = True
                p.requires_grad = trainable
    return model


model = Model()
model = load_pretrained_model(model)

model.to(device)
