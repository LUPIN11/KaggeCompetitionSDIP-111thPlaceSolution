from function import *
import pandas as pd

df1 = pd.read_csv('diffusiondb_12n.csv')
# df2 = pd.read_csv('/root/autodl-tmp/kaggle/img2text/input/gustavosta.csv')
# df3 = pd.read_csv('/root/autodl-tmp/kaggle/img2text/input/sd2.csv')
# df = pd.concat([df1, df2, df3])

prompts = [
    'hyper realistic photo of very friendly and dystopian crater',
    'ramen carved out of fractal rose ebony, in the style of hudson river school',
    'ultrasaurus holding a black bean taco in the woods, near an identical cheneosaurus',
    'a thundering retro robot crane inks on parchment with a droopy french bulldog',
    'portrait painting of a shimmering greek hero, next to a loud frill-necked lizard',
    'an astronaut standing on a engaging white rose, in the midst of by ivory cherry blossoms', ]

target_prompts = list(df1['prompt'])[:5000]
selected_prompts = []
for prompt in prompts:
    for target_prompt in target_prompts:
        cs = semantic_similarity(prompt, target_prompt)
        print(cs)
        if cs >= 0.8:
            print(target_prompt)
            selected_prompts.append(target_prompt)
print(len(selected_prompts))
