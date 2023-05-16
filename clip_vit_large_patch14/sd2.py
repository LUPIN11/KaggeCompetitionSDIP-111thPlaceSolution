import gc
import torch
import pandas as pd
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

# config
###################
prompt_path = 'prompts1_b.csv'
tar_dir = '/root/autodl-tmp/kaggle/PromptPredict/images/1_b/'
csv_path = 'pis1_b.csv'


###################

class StableDiffusion:
    def __init__(self, model_id='/root/autodl-tmp/kaggle/PromptPredict/sd2/kaggle/working/pipe'):
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