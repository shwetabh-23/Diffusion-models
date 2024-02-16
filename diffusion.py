from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
import torch.nn as nn
import pytorch_lightning as pl
from modules import Unet
from data_processing import get_data
from torch.utils.data import random_split, DataLoader

class Diffusion(pl.LightningModule):
    
    def __init__(self, noise_steps = 1000, beta_start = 1e-4, beta_end = 0.02, img_size = 256):
        super().__init__()
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.unet = Unet()
        
        self.beta = self.prepare_noise_schedule()
        #breakpoint()
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim = 0)

        self.lr = 1e-5
        self.epochs = 500

        self.dataset = get_data(dataset_path= r'data/')

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None].to('cuda')
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None].to('cuda')
        eps = torch.randn_like(x).to('cuda')
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps
    
    def sample_timesteps(self, n):
        return torch.randint(low= 1, high= self.noise_steps, size= (n, ))
    
    def sample(self, model, n):

        model.eval()
    
    def forward(self, x, t):
       #breakpoint()
       return self.unet(x, t)
    
    def configure_optimizers(self):
        #breakpoint()
        optim = torch.optim.AdamW(self.parameters(), self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size= self.epochs // 2, gamma = 0.1)

        return [optim], [scheduler]
    
    def criterion(self, X, X_hat):
        loss_fn = nn.MSELoss()
        loss = loss_fn(X, X_hat)
        return loss
    
    def generate_complete_dataset(self):
        #dataset = (self.dataset)
        train_size = int(0.5 * len(self.dataset))
        val_size = int(0.3 * len(self.dataset))
        test_size = int(0.2 * len(self.dataset))

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        self.generate_complete_dataset()
        return DataLoader(self.train_dataset, batch_size= 4, shuffle= True)
    
    def val_dataloader(self):

        return DataLoader(self.val_dataset, batch_size= 8, shuffle= False)
    
    def test_dataloader(self):

        return DataLoader(self.test_dataset, batch_size= 8, shuffle= False)
    

    def training_step(self, batch):
        X, y = batch
        t = self.sample_timesteps(X.shape[0])
        x_t, noise = self.noise_images(X, t)
        x_t = self(X, t)

        loss = self.criterion(X, x_t)
        return loss
    
    def validation_step(self, batch):
        X, y = batch
        t = self.sample_timesteps(X.shape[0])
        x_t, noise = self.noise_images(X, t)
        x_t = self(X, t)

        loss = self.criterion(X, x_t)
        return loss


if __name__ == '__main__':
    diffusion = Diffusion()
    breakpoint()

        
    
    