import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append('/root/autodl-tmp/kaggle/sentence-transformers')
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("/root/autodl-tmp/kaggle/all-MiniLM-L6-v2")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv('your_data.csv')
X = encoder.encode(df["prompt"].to_numpy(), batch_size=512, show_progress_bar=True)
y = df['similarity']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Train set size: {len(X_train)}, Val set size: {len(X_val)}')


class CFG:
    num_epochs = 100
    lr = 0.01
    batch_size = 16
    total_train_samples = len(X_train)
    total_val_samples = len(X_val)
    num_train_batches = int(total_train_samples / batch_size)
    num_val_batches = int(total_val_samples / batch_size)


X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size)


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, input_size * 3),
            nn.ReLU(),
            nn.Linear(input_size * 3, input_size * 3),
            nn.ReLU(),
            nn.Linear(input_size * 3, input_size),
            nn.ReLU(),
            nn.Linear(input_size, output_size)
        )

    def forward(self, x):
        return self.fc_layers(x)


model = Model(384, 1)
model.to(device)
criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=CFG.lr)
optimizer = optim.Adam(model.parameters(), lr=CFG.lr)

for epoch in range(CFG.num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    for X, y in train_loader:
        inputs = X.to(device)
        targets = y.to(device)

        outputs = model(inputs)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    for X, y in val_loader:
        with torch.no_grad():
            val_outputs = model(X.to(device))
            val_loss = criterion(val_outputs, y.to(device)).item()
            val_loss += val_loss.item()

    train_loss /= CFG.num_train_batches
    val_loss /= CFG.num_val_batches
    print(f'Epoch [{epoch}/{CFG.num_epochs}], Train Loss: {train_loss:.4}, Val Loss: {val_loss:.4}')
