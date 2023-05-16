import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
sys.path.append('/root/autodl-tmp/kaggle/img2text/input/sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



def get_train_test_split():
    sentence_encoder = SentenceTransformer(
        "/root/autodl-tmp/kaggle/img2text/input/sentence-transformers-222/all-MiniLM-L6-v2")
    df = pd.read_csv('resultOnTrain0.csv')
    X = df['prompt']
    X = sentence_encoder.encode(X.to_list(), batch_size=512, show_progress_bar=True, device="cuda")
    y = df['similarity'].map(lambda x: 0 if x < 0.5 else 1).to_list()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)
    print(f"train set size: {len(X_train)}, valid set size: {len(X_valid)}")
    return X_train, X_valid, y_train, y_valid


X_train, X_valid, y_train, y_valid = get_train_test_split()
weight0 = len(y_train)/(2*(len(y_train)-sum(y_train)))
weight0 *= 0.1
weight1 = len(y_train)/(2*sum(y_train))
print(f'{weight0} for {len(y_train)-sum(y_train)} samples')
print(f'{weight1} for {sum(y_train)} samples')
weight = torch.FloatTensor([weight0, weight1]).to(device)
criterion = nn.CrossEntropyLoss(weight=weight)

k = 0.9
print(k)
class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return X, y


class CFG:
    num_epochs = 50
    lr = 0.001
    batch_size = 32
    total_train_samples = len(y_train)
    total_val_samples = len(y_valid)
    num_train_batches = int(total_train_samples / batch_size)
    num_val_batches = int(total_val_samples / batch_size)


train_dataloader = DataLoader(dataset=Dataset(X_train, y_train),
                              batch_size=CFG.batch_size, shuffle=True, num_workers=8)
valid_dataloader = DataLoader(dataset=Dataset(X_valid, y_valid),
                              batch_size=CFG.batch_size, shuffle=False, num_workers=8)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)


model = Net()
model.to(device)

# optimizer = optim.SGD(model.parameters(), lr=CFG.lr)
optimizer = optim.Adam(model.parameters(), lr=CFG.lr)


bst_acc = 0
for epoch in range(CFG.num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    model.train()
    for X, y in tqdm(train_dataloader, leave=False):
        X = X.to(device)
        y = y.to(device)
        y_p = model(X)
        loss = criterion(y_p, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss

    model.eval()
    num_p0 = 0
    num_pt0 = 0
    for X, y in tqdm(valid_dataloader, leave=False):
        with torch.no_grad():
            X = X.to(device)
            y = y.to(device)

            y_p = model(X)
            loss = criterion(y_p, y)
            val_loss += loss

            # result = y_p.argmax(dim=1)
            # pred20 = torch.eq(result, 0)
            pred20 = y_p[:, 0] >= k
            pred20 = torch.where(pred20, torch.tensor([1]).to(device), torch.tensor([0]).to(device))
            true0 = torch.eq(y, 0)
            pt0 = pred20*true0
            num_p0 += sum(pred20)
            num_pt0 += sum(pt0)

    train_loss /= CFG.num_train_batches
    val_loss /= CFG.num_val_batches
    print(f'Epoch [{epoch}/{CFG.num_epochs}], Train Loss: {train_loss:.4}, Val Loss: {val_loss:.4}, Acc: {num_pt0/num_p0:.4}, Count: {num_pt0}')
    if num_pt0/num_p0 > bst_acc:
        bst_acc = num_pt0/num_p0
        torch.save(model.state_dict(), f'./regression/prompt_predict_model.pth')
        print('current model saved')


