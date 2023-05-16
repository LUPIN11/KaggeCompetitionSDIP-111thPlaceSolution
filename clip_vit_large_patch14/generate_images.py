import gc
import torch
import pandas as pd
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

# config
###################
prompt_path = 'prompts1.csv'
tar_dir = '/root/autodl-tmp/kaggle/PromptPredict/images/1/'
csv_path = 'pis1.csv'


###################

class StableDiffusion:
    def __init__(self, model_id="stabilityai/stable-diffusion-2"):
        self.scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=self.scheduler,
                                                            torch_dtype=torch.float16,
                                                            ).to("cuda")

    def generate(self, prompt):
        torch.cuda.empty_cache()
        gc.collect()
        image = self.pipe(prompt).images[0]
        return image


sd2 = StableDiffusion()
df = pd.read_csv(prompt_path)
prompts = df['prompt']
print(f'{len(prompts)} images to generate')
images = []
idx = 0

for prompt in prompts:
    image = sd2.generate(prompt)
    image_name = f'{idx}.png'
    idx += 1
    image_path = tar_dir + image_name
    image.save(image_path)
    images.append(image_path)

df = df.assign(filepath=images)
df.to_csv(f'{csv_path}')
print(len(df))
print(df.columns)
