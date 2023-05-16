import torch
from torch import nn


class ClipModel(torch.nn.Module):
    def __init__(self, k):
        super(ClipModel, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
        )
        self.k = k

    def forward(self, x):
        output = self.fc(x)
        output = output * self.k + x
        return output


class TransformerModel(torch.nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.layers = nn.Sequential(
            torch.nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            torch.nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            torch.nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 384),
        )

    def forward(self, x):
        output = self.layers(x)
        return output


class Net(torch.nn.Module):
    def __init__(self, clip_model_path=None, trans_model_path=None, k=1):
        super(Net, self).__init__()
        clip_model = ClipModel(k=k)
        trans_model = TransformerModel()
        if clip_model_path is not None:
            clip_model.load_state_dict(torch.load(clip_model_path))
        if trans_model_path is not None:
            trans_model.load_state_dict(torch.load(trans_model_path))
        self.clip_model = clip_model
        self.trans_model = trans_model
        self.fc1 = nn.Sequential(
            nn.Linear(512, 512),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(384, 384),
        )

    def forward(self, x):
        output = self.clip_model(x)
        output = self.fc1(output)
        output = self.trans_model(output)
        output = self.fc2(output)
        return output


class DirModel(torch.nn.Module):
    def __init__(self):
        super(DirModel, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Linear(768, 768),
            torch.nn.ReLU(),
            torch.nn.Linear(768, 768),
        )

    def forward(self, x):
        output = self.fc(x)
        norm = torch.norm(output)
        output = output / norm
        return output



if __name__ == "__main__":
    model = Net('./ckpt/512to512.pth', './ckpt/512to384.pth', 1)
    torch.save(model.state_dict(), 'mm_model.pth')
