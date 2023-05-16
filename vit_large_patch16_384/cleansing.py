import os
import unicodedata
import pandas as pd
from tqdm.notebook import tqdm


def is_english_only(string):
    for s in string:
        cat = unicodedata.category(s)
        if not cat in ['Ll', 'Lu', 'Nd', 'Po', 'Pd', 'Zs']:
            return False
    return True


df = pd.read_parquet('/root/autodl-tmp/kaggle/img2text/input/diffusiondb2m/metadata.parquet',
                     columns=['image_name', 'prompt', 'width', 'height'])
df['prompt'] = df['prompt'].str.strip()
df = df.dropna(subset=['prompt'])
df = df[~df['prompt'].str.contains('^(?:\s*|NULL|null|NaN)$', na=True)]
df = df[df['prompt'].apply(is_english_only)]
df.drop_duplicates(subset='prompt', inplace=True)
df.reset_index(drop=True, inplace=True)

df = df[['prompt']]
print(len(df))
df.to_csv('prompts150k.csv')

# # # df = df[(df['width'] == 512) & (df['height'] == 512)]
# df = df[df['width'] == df['height']]
# df['prompt'] = df['prompt'].str.strip()
# df = df.dropna(subset=['prompt'])
# df = df[df['prompt'].map(lambda x: 5 <= len(x.split())) <= 70]
# df = df[~df['prompt'].str.contains('^(?:\s*|NULL|null|NaN)$', na=True)]
# df = df[df['prompt'].apply(is_english_only)]
# # # df['head'] = df['prompt'].str[:15]
# # # df['tail'] = df['prompt'].str[-15:]
# # # df.drop_duplicates(subset='head', inplace=True)
# # # df.drop_duplicates(subset='tail', inplace=True)
# df.drop_duplicates(subset='prompt', inplace=True)
# df.reset_index(drop=True, inplace=True)
#
# for i in tqdm(range(1, 2000, 100)):
#     image_dir = f'/root/autodl-tmp/kaggle/img2text/input/diffusiondb2m/diffusiondb-2m-part-{str(i).zfill(4)}-to-{str(i + 99).zfill(4)}-of-2000/'
#     images = os.listdir(image_dir)
#     df.loc[df['image_name'].isin(images), 'filepath'] = image_dir + df['image_name']
# df = df[['filepath', 'prompt']].copy()
# assert not df['filepath'].isnull().any()
#
# print(len(df))
# # df.to_csv('ddb600k.csv', index=False)  # 保留了prompt重复和image大小不是512x512的图像
# # df = pd.read_csv('train2.csv')
# # print(len(df))
# # df = df.dropna(subset=['prompt'])
# # print(len(df))
# df.to_csv('diffusiondb.csv')
#
# # 放弃学习过长的prompt，因为sd2根本不可能把握住长句子中的每个要素
# # 而且长句子可能有害，因为长句子的要素太多，长句子输入图像这一步就会损失很多信息，我们不应该把这部分必然损失的信息作为对模型的惩罚
