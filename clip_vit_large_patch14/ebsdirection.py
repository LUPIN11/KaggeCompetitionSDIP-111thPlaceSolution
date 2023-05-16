import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
from timm.utils import AverageMeter
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-4
batch_size = 32
num_epochs = 100

ebs_i = np.load('image_features_L.npy')
print(len(ebs_i))
ebs_t = np.load('text_features_L.npy')
ebs_d = ebs_t - ebs_i



# print(ebs_in.shape, ebs_out.shape)
# print(np.linalg.norm(ebs_in[0]))

ebs_in_train, ebs_in_valid, ebs_out_train, ebs_out_valid = train_test_split(ebs_i, ebs_d, test_size=0.1,
                                                                            random_state=42)


class EbDataset(Dataset):
    def __init__(self, ebs_in, ebs_out):
        self.ebs_in = ebs_in
        self.ebs_out = ebs_out

    def __len__(self):
        return self.ebs_in.shape[0]

    def __getitem__(self, item):
        return self.ebs_in[item], self.ebs_out[item]


train_dataloader = DataLoader(EbDataset(ebs_in_train, ebs_out_train),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)
valid_dataloader = DataLoader(EbDataset(ebs_in_valid, ebs_out_valid),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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


model = Net()
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=lr)
optimizer.zero_grad()

# criterion = torch.nn.MSELoss().to(device)
# cosine_loss = torch.nn.CosineEmbeddingLoss(margin=0.95).to(device)
cosine_sim = torch.nn.CosineSimilarity(dim=1).to(device)

bst_cos = -1

for epoch in range(num_epochs):
    train_meters = {'loss': AverageMeter(),
                    'cos': AverageMeter()}
    valid_meters = {'cos': AverageMeter()}
    model.train()
    bar = tqdm(train_dataloader, leave=False)
    for X, y in bar:
        X, y = X.to(device), y.to(device)
        output = model(X)
        # loss = criterion(output, y)
        loss = -cosine_sim(output, y).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_meters['loss'].update(loss.item(), n=X.shape[0])
        train_meters['cos'].update(cosine_sim(output, y).mean().item(), n=X.shape[0])
        bar.set_postfix(trn_loss=f'{train_meters["loss"].avg:.4f}', trn_cos=f'{train_meters["cos"].avg:.4f}')
    print(f"Epoch {epoch + 1:d} / train_cos={train_meters['cos'].avg:.4f}")
    model.eval()
    bar = tqdm(valid_dataloader, leave=False)
    for X, y in bar:
        X, y = X.to(device), y.to(device)
        output = model(X)
        valid_meters['cos'].update(cosine_sim(output, y).mean().item(), n=X.shape[0])
        bar.set_postfix(val_cos=f'{valid_meters["cos"].avg:.4f}')
    print(f"Epoch {epoch + 1:d} / valid_cos={valid_meters['cos'].avg:.4f}")
    if valid_meters['cos'].avg > bst_cos:
        bst_cos = valid_meters['cos'].avg
        torch.save(model.state_dict(), f'./ckpt/dir.pth')
        print('current model saved')
