# %pip install faiss-gpu

import sys
import re
import faiss
import torch
import numpy as np
import polars as pl
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append('./input/sentence-transformers-222/sentence-transformers')
from sentence_transformers import SentenceTransformer


def check_string(string: str) -> bool:
    # Checks if the given string contains any character other than alphanumeric characters, comma, dot, hyphen or whitespace
    return bool(re.search(r'[^A-Za-z0-9,.\\-\\s]', string))


# Load data from a Parquet file
# For the purpose of illustration, the amount of data will be reduced
pldf = pl.read_parquet("/kaggle/input/diffusiondb-metadata/metadata.parquet",
                       columns=['image_name', 'prompt', 'width', 'height'])
# df = pd.read_csv('/root/autodl-tmp/kaggle/img2text/input/data.csv')
# print(df.columns)
# pldf = pl.from_pandas(df)

# Select only those prompts that have five or more words
pldf = pldf.filter(pl.col("prompt").str.split(" ").apply(lambda x: 5 <= len(x) <= 77))

# Select only those prompts that are not blank, NULL, null, or NaN
pldf = pldf.filter(~pl.col("prompt").str.contains('^(?:\s*|NULL|null|NaN)$'))

pldf = pldf.filter(pl.col("prompt").apply(check_string))
print('after rule baesd filter', len(pldf))  # 1358948 1282476

# For the purpose of illustration, we will reduce the amount of data
# pldf = pldf[:100000]

model = SentenceTransformer("./input/sentence-transformers-222/all-MiniLM-L6-v2")
vector = model.encode(pldf["prompt"].to_numpy(), batch_size=512, show_progress_bar=True, device="cuda",
                      convert_to_tensor=True)

threshold = 0.9  # Set the threshold for similarity.
n_neighbors = 1000  # Set the number of neighbors to consider.

# Perform batch processing because processing all data at once may cause resource shortage.
batch_size = 1024  # Set the batch size (i.e., the number of data items to be processed at once).
similar_vectors = []  # Create an empty list to store similar vectors.

# Create an IndexFlatIP index using the Faiss library
# The term 'IP' represents the Inner Product,
# which is equivalent to cosine similarity as it involves taking the dot product of normalized vectors.
index = faiss.IndexFlatIP(384)

# Normalize the input vector and add it to the IndexFlatIP
index.add(F.normalize(vector).cpu().numpy())

for i in tqdm(range(0, len(vector), batch_size)):
    # Get the target batch for processing.
    batch_data = vector.cpu().numpy()[i:i + batch_size]
    # Neighborhood search based on cosine similarity.
    similarities, indices = index.search(batch_data, n_neighbors)

    # Extract indexes and similarities of data to be deleted.
    for j in range(similarities.shape[0]):
        close_vectors = indices[j, similarities[j] >= threshold]
        index_base = i
        # Get only the similar vectors that exclude itself
        close_vectors = close_vectors[close_vectors != index_base + j]
        similar_vectors.append((index_base + j, close_vectors))

pldf = pldf.with_columns(pl.Series(values=list(range(len(pldf))), name="index"))
pldf = pldf.filter(~pl.col("index").is_in(np.unique(np.concatenate([x for _, x in similar_vectors])).tolist()))

# for i, _ in tqdm(enumerate(range(1, 2000, 100)), total=20):
#     image_dir = Path("/kaggle/input/diffusiondb-2m-part-{:04d}-to-{:04d}-of-2000/".format(i * 100 + 1, (i + 1) * 100))
#     pldf = pldf.with_columns(
#         pl.when(pl.col("image_name").is_in([str(file_path.name) for file_path in image_dir.glob("*.png")]))
#         .then(str(image_dir) + "/" + pl.col("image_name"))
#         .otherwise(pl.col("image_name"))
#         .alias("image_name")
#     )
print(len(pldf))
newpdf = pldf.to_pandas()
newpdf['prompt'] = newpdf['prompt'].apply(lambda x: re.sub(r"(\d)\s", r'\1', x))
newpdf['prompt'] = newpdf['prompt'].apply(lambda x: re.sub(r'\s(-)\s', r'\1', x))
pldf = pl.from_pandas(newpdf)
print(newpdf.columns)
pldf.select(pl.col("filepath", "prompt")).write_csv("test.csv")
pldf.select(pl.col("filepath", "prompt")).head()
