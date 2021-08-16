import torch
import torch.nn as nn

from models import BaseModel
from models.utils import *

class Discriminator(BaseModel):

    def __init__(self, ndf, nc, z_dim):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.nc = nc
        self.z_dim = z_dim

        self.FE = nn.Sequential(
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(ndf, ndf*2, kernel_size= 4, stride = 2, padding=1),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(),

            nn.Conv2d(ndf*2, ndf*4, kernel_size = 4, stride=2, padding=1),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(),

            nn.Conv2d(ndf*4 , ndf*4, kernel_size=3, stride =1, padding=1),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(),

            nn.Conv2d(ndf*4 , ndf*4, kernel_size=3, stride =1, padding=1),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(),

            nn.Conv2d(ndf*4 , ndf*16, kernel_size=9, stride =1, padding=4),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU()
        )

        self.q = nn.Sequential(
            nn.Linear(ndf*16*8*8, z_dim)
        )

        self.d = nn.Sequential(
            nn.Linear(ndf*16*8*8, 1)
        )

        self.weight_init()
    
    def add_guassian_noise(self, device, input, global_iter, noise_iter):
        B, C, H, W = input.size()
        noise_factor = 1 - (global_iter / noise_iter)
        noise = torch.randn(input.size()).to(device)
        return torch.add(input,noise, alpha= noise_factor)  

    def forward_q(self, device, input, global_iter, noise_iter):
        B, C, H, W = input.size()
        new_input = self.add_guassian_noise(device, input, global_iter, noise_iter)
        feature = self.FE(new_input)
        feature = feature.view(B, -1)
        z_rec = self.q(feature)
        return z_rec

    def forward_d(self, device, input, global_iter, noise_iter):
        B, C, H, W = input.size()
        new_input = self.add_guassian_noise(device, input, global_iter, noise_iter)
        feature = self.FE(new_input)
        feature = feature.view(B, -1)
        out = self.d(feature)
        return out
    
    def forward(self, device, input, global_iter, noise_iter):    
        B, C, H, W = input.size()
        new_input = self.add_guassian_noise(device, input, global_iter, noise_iter)
        feature = self.FE(new_input)
        feature = feature.view(B, -1)
        z_rec = self.q(feature)
        out = self.d(feature)
        return [z_rec, out]



