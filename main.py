import torch
import torchvision
import pytorch_lightning as pl
from diffusion import Diffusion

model = Diffusion()
trainer = pl.Trainer(max_epochs= 5, accelerator= 'cuda', enable_progress_bar= True)
trainer.fit(model=model)