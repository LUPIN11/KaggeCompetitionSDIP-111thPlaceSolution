import unicodedata
import pandas as pd
import os
from tqdm.notebook import tqdm

# df = pd.read_parquet('/root/autodl-tmp/kaggle/img2text/input/diffusiondb2m/metadata.parquet',
#                      columns=['image_name', 'prompt', 'width', 'height'])
#
#
# for i in tqdm(range(1, 2000, 100)):
#     image_dir = f'/root/autodl-tmp/kaggle/img2text/input/diffusiondb2m/diffusiondb-2m-part-{str(i).zfill(4)}-to-{str(i + 99).zfill(4)}-of-2000/'
#     images = os.listdir(image_dir)
#     df.loc[df['image_name'].isin(images), 'filepath'] = image_dir + df['image_name']
#
# df = df[['filepath', 'prompt']].copy()
# assert not df['filepath'].isnull().any()
# print(len(df))
# df.to_csv('diffusiondb2m.csv', index=False)
# /kaggle/input/gustavosta-stable-diffusion-prompts-sd2-v2/eval_images/00000000.jpg
# /kaggle/input/gustavosta-stable-diffusion-prompts-SD2/eval_images/00000000.jpg
# /root/autodl-tmp/kaggle/img2text/input/gustavosta-stable-diffusion-prompts-SD2
# /root/autodl-tmp/kaggle/img2text/input/gustavosta-stable-diffusion-prompts-sd2-v2
# df0 = pd.read_csv('./input/gustavosta-stable-diffusion-prompts-SD2/eval.csv')
# df1 = pd.read_csv('./input/gustavosta-stable-diffusion-prompts-SD2/train.csv')
# df2 = pd.read_csv('./input/gustavosta-stable-diffusion-prompts-sd2-v2/eval.csv')
# df3 = pd.read_csv('./input/gustavosta-stable-diffusion-prompts-sd2-v2/train.csv')
# df = pd.concat([df0, df1, df2, df3], axis=0)
# print(len(df))
#
#
# def new_path(path):
#     elements = path.split('/')
#     return '/root/autodl-tmp/kaggle/img2text/' + '/'.join(elements[-4:])
#
#
# df['image_path'] = df['image_path'].map(new_path)
# df = df.rename(columns={'Prompt': 'prompt', 'image_path': 'filepath'})
# print(df.head())
# df.to_csv('gustavosta.csv', index=False)

# df1 = pd.read_csv('./input/sd2gpt2/gpt_generated_prompts.csv')
# print(len(df1))
# # /root/autodl-tmp/kaggle/img2text/input/sd2gpt2/gpt_generated_images/gpt_generated_images
# count = 0
#
#
# def get_filepath(index_value):
#     global count
#     image_dir = '/root/autodl-tmp/kaggle/img2text/input/sd2gpt2/gpt_generated_images/gpt_generated_images'
#     count += 1
#     return os.path.join(image_dir, str(count - 1) + '.png')
#
#
# # 将该函数应用于 DataFrame 的索引列，生成 'filepath' 列
# df1['filepath'] = df1['prompt'].map(get_filepath)
#
# # print(df1.iloc[0]['filepath'])
#
# # autodl-tmp/kaggle/img2text/input/sd2hardcode/hardcoded_images/hardcoded_images
# df2 = pd.read_csv('./input/sd2hardcode/hardcoded_prompts.csv')
# cou = 0
# print(len(df2))
#
# def get_filepa(index_value):
#     global cou
#     image_dir = '/root/autodl-tmp/kaggle/img2text/input/sd2hardcode/hardcoded_images/hardcoded_images'
#     cou += 1
#     return os.path.join(image_dir, str(cou - 1) + '.png')
#
#
# df2['filepath'] = df2['prompt'].map(get_filepa)
# df = pd.concat([df1, df2])
# print(len(df))
# df.to_csv('sd2.csv')

#
# df1 = pd.read_csv('/root/autodl-tmp/kaggle/img2text/input/diffusiondb.csv')
# print(df1.columns)
# df1 = df1.drop(columns=['Unnamed: 0'])
# df1.to_csv('diffusiondb.csv')
# df2 = pd.read_csv('/root/autodl-tmp/kaggle/img2text/input/gustavosta.csv')
# df3 = pd.read_csv('/root/autodl-tmp/kaggle/img2text/input/sd2.csv')
# print(df1.columns)
# print(df2.columns)
# print(df3.columns)
# df = pd.concat([df1, df2, df3])
# print(df.columns)
# print(len(df3))
# df.to_csv('data.csv', index=False)

# df = pd.read_csv('new_valid.csv')
# mean_similarity = df['similarity'].mean()
# print(mean_similarity)
df1 = pd.read_csv('train1.csv')
df1 = df1.dropna(subset='filepath')
# df2 = pd.read_csv('train.csv')
print(len(df1))
# print(len(df2))
