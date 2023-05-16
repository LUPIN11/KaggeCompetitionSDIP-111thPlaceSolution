from random import choice, randint
from copy import deepcopy
from matplotlib import pyplot as plt
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from function import *
import pandas as pd
from tqdm import tqdm


class StableDiffusion:
    def __init__(self, model_id="stabilityai/stable-diffusion-2"):
        self.scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=self.scheduler,
                                                            torch_dtype=torch.float16,
                                                            ).to("cuda")

    def generate(self, prompt):
        torch.cuda.empty_cache()
        return self.pipe(prompt).images


def load_vocab(txt_path):
    with open(txt_path, 'r') as file:
        words_list = [line.strip() for line in file]
        words_set = set(words_list)
    return words_list, words_set


sd2 = StableDiffusion()
nouns_list, nouns_set = load_vocab('./nouns.txt')
adjs_list, adjs_set = load_vocab('./adjectives.txt')
verbs_list, verbs_set = load_vocab('./verbs.txt')
df2save = pd.DataFrame(columns=['prompt', 'filepath'])
img_idx = 0


def replace_a_word(elements):
    new_elements = deepcopy(elements)
    for _ in range(2 * len(elements)):
        i = randint(0, len(elements) - 1)
        if elements[i] in nouns_set or elements[i][:-1] in nouns_set or elements[i][:-2] in nouns_set:
            new_e = choice(nouns_list)
        elif elements[i] in adjs_set:
            new_e = choice(adjs_list)
        elif elements[i] in verbs_set or elements[i][:-1] in verbs_set or elements[i][:-2] in verbs_set or elements[i][
                                                                                                           :-3] in verbs_set:
            new_e = choice(verbs_list)
        else:
            continue
        new_elements[i] = new_e
        return new_elements
    return new_elements  # replace失败，原封不动返回


def gen_prompts(prompt, max_its, steps=None):
    new_prompts = []
    elements = prompt.split()
    if steps is None:
        steps = int(len(elements) / 2) if len(elements) % 2 == 0 else int((len(elements) - 1) / 2)
    for _ in range(max_its):
        new_elements = deepcopy(elements)
        for _ in range(steps):
            new_elements = replace_a_word(new_elements)
        new_prompts.append(' '.join(new_elements))
    return new_prompts


def gen_image(prompts):
    global df2save, img_idx
    images = sd2.generate(prompts)
    for i in range(len(prompts)):
        image = images[i]
        prompt = prompts[i]
        new_image_name = str(img_idx) + '.jpg'
        img_idx += 1
        new_image_path = '/root/autodl-tmp/kaggle/img2text/images/1/' + new_image_name
        image.save(new_image_path)
        new_row = {'prompt': prompt, 'filepath': new_image_path}
        df2save = pd.concat([df2save, pd.DataFrame(new_row, index=[0])], ignore_index=True)


if __name__ == "__main__":
    try:
        # df = pd.read_csv('result6748.csv')
        # df_sorted = df.sort_values(by='similarity', ascending=True)
        # df_min_similarity = df_sorted.head(10000)
        # prompts = list(df_min_similarity['prompt'])
        prompts = [
            'hyper realistic photo of very friendly and dystopian crater',
            'ramen carved out of fractal rose ebony, in the style of hudson river school',
            'ultrasaurus holding a black bean taco in the woods, near an identical cheneosaurus',
            'a thundering retro robot crane inks on parchment with a droopy french bulldog',
            'portrait painting of a shimmering greek hero, next to a loud frill-necked lizard',
            'an astronaut standing on a engaging white rose, in the midst of by ivory cherry blossoms',
        ]
        new_prompts = []
        for prompt in prompts:
            new_prompts.extend(gen_prompts(prompt, 1, 1))
        for i in range(0, len(new_prompts), 2):
            nps = new_prompts[i:i + 2]  # 取出当前子列表
            gen_image(nps)
    finally:
        df2 = pd.read_csv('train.csv')
        df2save = pd.concat([df2, df2save], ignore_index=True)
        df2save.to_csv('train1.csv', index=False)
    # prompts = [
    #     'hyper realistic photo of very friendly and dystopian crater',
    #     'ramen carved out of fractal rose ebony, in the style of hudson river school',
    #     'ultrasaurus holding a black bean taco in the woods, near an identical cheneosaurus',
    #     'a thundering retro robot crane inks on parchment with a droopy french bulldog',
    #     'portrait painting of a shimmering greek hero, next to a loud frill-necked lizard',
    #     'an astronaut standing on a engaging white rose, in the midst of by ivory cherry blossoms',
    # ]
    # sum_sim = 0
    # count = 0
    # for prompt in prompts:
    #     print('p')
    #     new_prompts = gen_prompts(prompt, 5, 4)
    #     for np in new_prompts:
    #         print('np')
    #         sim = semantic_similarity(np, prompt)
    #         sum_sim += sim
    #         count += 1
    # print(sum_sim / count)
# 1step 0.8618
# 2step 0.8086
# 3step 0.6838
# 4step 0.6391
# 5step 0.5955
# 10step 0.3649
# 20step 0.2435