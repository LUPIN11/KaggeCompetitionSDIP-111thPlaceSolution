import sys
import faiss
import pandas as pd

sys.path.append('/root/autodl-tmp/kaggle/img2text/input/sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer
from functions import *
from tqdm import tqdm
import torch.nn.functional as F
import gc

encoder = SentenceTransformer("/root/autodl-tmp/kaggle/img2text/input/sentence-transformers-222/all-MiniLM-L6-v2")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# config
####################
# prompt_path = 'new_prompts_gpt.csv'
# prompt_path = '/root/autodl-tmp/kaggle/img2text/input/sd2.csv'
save_path = 'prompts1.csv'
base_path = '../PromptPredict/train0.csv'
num_iters = 15000  # gpt2 generate num_iters*10 prompts
k = 0.9
model_path = '/root/autodl-tmp/kaggle/PromptPredict/regression/prompt_predict_model.pth'
####################

# generate prompts
# from transformers import pipeline
#
# model = pipeline('text-generation', model='./gpt2')
#
#
# def generate_prompts(core="try imagine this imaginative picture: ", num_prompts=10):
#     return model(
#         core,
#         max_length=60,
#         num_return_sequences=num_prompts,
#         temperature=0.70,
#         eos_token_id=model.tokenizer.convert_tokens_to_ids("."),
#         early_stopping=True,
#         top_k=800,
#         top_p=800,
#         pad_token_id=model.tokenizer.convert_tokens_to_ids(" "),
#     )
#
#
# candidates = []
# # count = 0
# print('Begin Generating')
# for _ in tqdm(range(num_iters)):
#     candidates.extend([sen['generated_text'] for sen in generate_prompts()])
# df = pd.DataFrame({'prompt': candidates})
# # df.to_csv('new_prompts2.csv')
# # df = pd.read_csv(prompt_path)[['prompt']]
#
# del candidates, model, generate_prompts
# gc.collect()
#
# print('generate ', len(df), 'prompts')
# # print(df.columns)
# df['prompt'] = df['prompt'].apply(lambda x: ' '.join(remove_prefix(sentence) for sentence in x.split('. \n')))
df = pd.read_csv('hardcodeprompts.csv')
# df['prompt'] = df['prompt'].apply(lambda x: ' '.join(remove_prefix(sentence) for sentence in x.split('. \n')))
X = encoder.encode(df["prompt"].to_numpy(), batch_size=512, show_progress_bar=True)


# FLITER 1:
print('Begin Filter 1')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
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
model.load_state_dict(torch.load(model_path))
model.to(device)

X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# 预测并保存结果
model.eval()
with torch.no_grad():
    preds = model(X_tensor)
    p0 = preds[:, 0] >= k
df = df[p0.cpu().numpy()]
print('after filter1:', len(df))
del X, X_tensor, model, Net, preds, p0
torch.cuda.empty_cache()
gc.collect()

# FLITER2:
batch_size = 1024
n_neighbors = 1000
vector = encoder.encode(df["prompt"].to_numpy(), batch_size=512, show_progress_bar=True, device="cuda",
                        convert_to_tensor=True)
similar_vectors = []
index = faiss.IndexFlatIP(384)
index.add(F.normalize(vector).cpu().numpy())
for i in tqdm(range(0, len(vector), batch_size)):
    batch_data = vector.cpu().numpy()[i:i + batch_size]
    similarities, indices = index.search(batch_data, n_neighbors)
    for j in range(similarities.shape[0]):
        close_vectors = indices[j, similarities[j] >= 0.95]
        index_base = i
        close_vectors = close_vectors[close_vectors != index_base + j]
        similar_vectors.append((index_base + j, close_vectors))

df['index'] = np.arange(len(df))
df = df[~df['index'].isin(np.unique(np.concatenate([x for _, x in similar_vectors])).tolist())]  # 筛选并去重

df = df.drop(columns=['index'])
print('after filter2:', len(df))

del vector, index, similar_vectors
torch.cuda.empty_cache()
gc.collect()

# FLITER3:
# 读取第一组vector
df1 = pd.read_csv(base_path)
print(f"Length of base: {len(df1)}")
vector1 = encoder.encode(df1["prompt"].to_numpy(), batch_size=batch_size, show_progress_bar=True, device="cuda",
                         convert_to_tensor=True)

# 读取第二组vector
vector2 = encoder.encode(df["prompt"].to_numpy(), batch_size=batch_size, show_progress_bar=True, device="cuda",
                         convert_to_tensor=True)

index = faiss.IndexFlatIP(384)
index.add(F.normalize(vector1).cpu().numpy())

drop_vectors = []
for i in tqdm(range(0, len(vector2), batch_size)):
    batch_data = vector2.cpu().numpy()[i:i + batch_size]
    similarities, indices = index.search(batch_data, n_neighbors)
    for j in range(similarities.shape[0]):
        close_vectors = indices[j, similarities[j] >= 0.95]
        index_base = i
        close_vectors = close_vectors[close_vectors != index_base + j]
        drop_vectors.append((index_base + j, close_vectors))

df['index'] = np.arange(len(df))
df = df[~df['index'].isin(np.unique(np.concatenate([x for _, x in drop_vectors])).tolist())]  # 筛选并去重
df = df.drop(columns=['index'])
print('after fliter3 1/2:', len(df))

del vector2, drop_vectors, df1
torch.cuda.empty_cache()
gc.collect()

vector2 = encoder.encode(df["prompt"].to_numpy(), batch_size=batch_size, show_progress_bar=True, device="cuda",
                         convert_to_tensor=True)

save_vectors = []
for i in tqdm(range(0, len(vector2), batch_size)):
    batch_data = vector2.cpu().numpy()[i:i + batch_size]
    similarities, indices = index.search(batch_data, n_neighbors)
    for j in range(similarities.shape[0]):
        close_vectors = indices[j, similarities[j] >= 0.4]
        index_base = i
        close_vectors = close_vectors[close_vectors != index_base + j]
        save_vectors.append((index_base + j, close_vectors))

df['index'] = np.arange(len(df))
df = df[df['index'].isin(np.unique(np.concatenate([x for _, x in save_vectors])).tolist())]  # 筛选并去重
df = df.drop(columns=['index'])
print('after fliter3', len(df))
print(len(df))
print(df.columns)
df.to_csv(save_path)
