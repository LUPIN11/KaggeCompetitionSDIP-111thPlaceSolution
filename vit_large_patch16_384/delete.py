import os
import pandas as pd

df1 = pd.read_csv('train.csv')
df2 = pd.read_csv('valid.csv')
print(len(df1))
print(len(df2))
df = pd.concat([df1, df2], ignore_index=True)
print(len(df))
keep_files = set(df['filepath'].unique().tolist())

folder_path = '/root/autodl-tmp/kaggle/img2text/input/diffusiondb2m'
count = 0
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        if os.path.isfile(file_path) and file_path not in keep_files:
            os.remove(file_path)
            count += 1
            print(count)
