import torch
from matplotlib import pyplot as plt
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler


class StableDiffusion:
    def __init__(self, model_id="stabilityai/stable-diffusion-2"):
        self.scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=self.scheduler,
                                                            torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")

    def generate(self, prompt, path=None, show=False):
        image = self.pipe(prompt).images[0]
        if path:
            image.save(path)
        if show:
            plt.imshow(image)
            plt.show()


if __name__ == "__main__":
    stable_diffusion = StableDiffusion()
    stable_diffusion.generate("a photo of an astronaut riding a horse on mars", show=True)
