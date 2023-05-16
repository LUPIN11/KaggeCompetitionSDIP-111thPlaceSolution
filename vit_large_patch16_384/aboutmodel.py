from torch import nn
from transformers import AutoModel


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
        print(name)
        if name == 'vision':
            for pn, p in child.named_parameters():
                if str(18) in pn:
                    """start unfreezing layer , the weights are trainable"""
                    trainable_model_weights = True
                p.requires_grad = trainable_model_weights
                if p.requires_grad:
                    print(f"{pn} is set to be trainable.")

    return model.to('cuda')

load_pretrained_model()