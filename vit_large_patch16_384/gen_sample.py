from random import choice, randint
from copy import deepcopy
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from function import *
import pandas as pd


class StableDiffusion:
    def __init__(self, model_id="stabilityai/stable-diffusion-2"):
        self.scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=self.scheduler,
                                                            torch_dtype=torch.float16,
                                                            ).to("cuda")

    def generate(self, prompt):
        torch.cuda.empty_cache()
        image = self.pipe(prompt).images[0]
        return image


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
    for prompt in prompts:
        if len(prompt) == 0:
            continue
        image = sd2.generate(prompt)
        new_image_name = str(img_idx) + '.jpg'
        img_idx += 1
        # todo images下的文件夹
        new_image_path = '/root/autodl-tmp/kaggle/img2text/images/5/' + new_image_name
        image.save(new_image_path)
        new_row = {'prompt': prompt, 'filepath': new_image_path}
        df2save = pd.concat([df2save, pd.DataFrame(new_row, index=[0])], ignore_index=True)


if __name__ == "__main__":
    try:
        df = pd.read_csv('p5.csv')
        prompts = list(df['prompt'])
        for prompt in prompts:
            new_prompts = gen_prompts(prompt, 1, 1)  # 1 step的平均波动甚至大于dalao的过滤阈值
            # new_prompts.append(prompt)
            # gen_image([prompt])  # 除非是到最后的顽固样本才在本身身上增加数据
            gen_image(new_prompts)
    finally:
        df2save.to_csv('new5.csv', index=False)

