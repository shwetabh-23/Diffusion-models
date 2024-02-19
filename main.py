import torch
import torchvision
import pytorch_lightning as pl
from diffusion import Diffusion
from pytorch_lightning.loggers import TensorBoardLogger
import os
import objgraph


model = Diffusion()
model_path = r'models/v2/model.pth'
logger = TensorBoardLogger(save_dir= r'models/', name= 'v2')
if not os.path.exists(model_path):
    trainer = pl.Trainer(max_epochs= 20, accelerator= 'cuda', enable_progress_bar= True, logger= logger)
    trainer.fit(model=model)
    torch.save(model.state_dict(), model_path)
state_dict = torch.load(model_path)
model.load_state_dict(state_dict=state_dict)
model = model.to('cpu')
trainer = pl.Trainer(logger=logger, accelerator= 'cuda')
#result = trainer.test(model=model)
sample = model.sample(n = 1)
breakpoint()
