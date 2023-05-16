import sys
ST_MODULE_PATH = './input/sentence-transformers-222/sentence-transformers'
sys.path.append(ST_MODULE_PATH)
ST_MODEL_PATH = './input/sentence-transformers-222/all-MiniLM-L6-v2'
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('diffusiondb_800k.csv')
print(len(df['prompt']))
model = SentenceTransformer(ST_MODEL_PATH, device='cpu')
embeddings = model.encode(df['prompt'].tolist(), show_progress_bar=False, convert_to_tensor=True)
similarity_matrix = cosine_similarity(embeddings)
new_df = df[similarity_matrix < 0.9]
print(len(new_df))
# new_df.to_csv('filtered_file.csv', index=False)