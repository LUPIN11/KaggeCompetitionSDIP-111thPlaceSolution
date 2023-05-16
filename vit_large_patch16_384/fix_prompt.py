# import pandas as pd
# import re

# df = pd.read_csv('valid.csv')
# print(len(df))
#
#
# # df['prompt'] = df['prompt'].apply(lambda x: re.sub(r"(\d)\s", r'\1', x))
# # df.to_csv('valid.csv')
# df['prompt'] = df['prompt'].astype(str)
# pattern = r'\d\s'
# contains_digit_space = df[df['prompt'].str.contains(pattern)]
# print(contains_digit_space)
# contains_digit_space.to_csv('see.csv')
# print('over')

import pandas as pd
import re

# df = pd.read_csv('dup_train.csv')
# print(len(df))


def clean_prompt(prompt):
    prompt = re.sub(r'(?<=\S)(?=\d)|(?<=\d)(?=\S)', ' ', prompt)
    prompt = re.sub(r'\s{2,}', ' ', prompt)
    return prompt


# df['prompt'] = df['prompt'].apply(clean_prompt)
# df.to_csv('train.csv')

s = 'abc123def45   6ghi789jkl,  1 9   30s, 89years, 1   man, 4  k ,   8k, 3d,   22mm, f12  ,   love22'
print(clean_prompt(s))


