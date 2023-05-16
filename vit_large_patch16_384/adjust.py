import pandas as pd
import re

# text = "dgdsgfh  1 9 9 0 s sgsgsg"
# pattern = r"(\d)\s"
# replacement = r"\1"
# new_text = re.sub(pattern, replacement, text)
#
# print(new_text)

# df = pd.read_csv('/root/autodl-tmp/kaggle/img2text/input/data.csv')
# df['prompt'] = df['prompt'].apply(lambda x: re.sub(r"(\d)\s", r'\1', x))
# df.to_csv('/root/autodl-tmp/kaggle/img2text/input/data.csv', index=False)


# text = "Hello , world !   How are you ? hot - girl"
# new_text = re.sub(r'\s(-)\s', r'\1', text)
# print(new_text)
import unicodedata

def is_english_only(string):
    for s in string:
        cat = unicodedata.category(s)
        if not cat in ['Ll', 'Lu', 'Nd', 'Po', 'Pd', 'Zs']:
            return False
    return True

s = 'ð ð°!!!!! abstract!! by atey ghailan and ( ( edward hopper ) ) bautiful!!! color city flat '

print(is_english_only(s))