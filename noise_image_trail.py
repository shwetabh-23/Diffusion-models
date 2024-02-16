import torch 
import torch.nn
import PIL
from diffusion import Diffusion
import numpy as np

def show_image(image):
    curr_image = image * 255
    curr_image = np.array(curr_image, dtype = np.uint8)
    transposed_image = np.transpose(curr_image, (1, 2, 0))
    x = PIL.Image.fromarray(transposed_image)
    x.show()

diffusion = Diffusion()
diffusion.generate_complete_dataset()
batch = next(iter(diffusion.train_dataloader()))[0]

image = batch[0]
t = diffusion.sample_timesteps(8)
t = torch.Tensor([10, 20, 30, 40, 500, 600, 700, 800, 900]).long()
show_image(image)
noise_image, _ = diffusion.noise_images(image, t)

for image in noise_image:
    breakpoint()
    show_image(image)
