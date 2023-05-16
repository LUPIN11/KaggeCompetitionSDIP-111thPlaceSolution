import pandas as pd
import torch
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
sys.path.append('/root/autodl-tmp/kaggle/img2text/input/sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer
from functions import *

encoder = SentenceTransformer("/root/autodl-tmp/kaggle/img2text/input/sentence-transformers-222/all-MiniLM-L6-v2")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 读取只有prompt列的数据
df = pd.read_csv('prompts0.csv')


print(len(df))
X = encoder.encode(df["prompt"].to_numpy(), batch_size=512, show_progress_bar=True)


# 加载模型

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
model.load_state_dict(torch.load('/root/autodl-tmp/kaggle/PromptPredict/regression/prompt_predict_model.pth'))
model.to(device)

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# 预测并保存结果
model.eval()
with torch.no_grad():
    preds = model(X_tensor)
    p0 = preds[:, 0] >= 0.999999
print(len(p0))
print(sum(p0))
# df = pd.DataFrame({'prompt': df['prompt'], 'similarity': preds})
# df.to_csv('hard_gpt_samples.csv')
# plt.hist(preds, bins=50, range=(0, 1))
# plt.show()
