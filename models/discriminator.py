import torch
import torch.nn as nn

from models import BaseModel
from models.utils import *

class Weight_Shared_Discriminator(BaseModel):

    def __init__(self, ndf, nc, z_dim, weight_init):
        super(Weight_Shared_Discriminator, self).__init__()
        self.ndf = ndf
        self.nc = nc
        self.z_dim = z_dim

        self.FE = nn.Sequential(
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf, ndf*2, kernel_size= 4, stride = 2, padding=1, bias = False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*2, ndf*4, kernel_size = 4, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2)

        )

        self.q = nn.Sequential(
            
            nn.Conv2d(ndf*4 , ndf*4, kernel_size=3, stride =1, padding=1, bias = False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*4 , ndf*4, kernel_size=3, stride =1, padding=1, bias = False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*4 , ndf*16, kernel_size=8, bias = False),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*16, z_dim, kernel_size = 1)
        )

        self.d = nn.Sequential(
            
            nn.Conv2d(ndf*4 , ndf*4, kernel_size=3, stride =1, padding=1, bias = False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*4 , ndf*4, kernel_size=3, stride =1, padding=1, bias = False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*4 , ndf*16, kernel_size=8, bias = False),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*16, 1, kernel_size= 1)
        )

        self.weight_init(weight_init)
    
    # def add_guassian_noise(self, device, input, global_iter, noise_iter):
    #     B, C, H, W = input.size()
    #     noise_factor = 1 - (global_iter / noise_iter)
    #     noise = torch.randn(input.size()).to(device)
    #     return torch.add(input,noise, alpha= noise_factor)  

    def forward_q(self,input):
        B, C, H, W = input.size()
        # new_input = self.add_guassian_noise(device, input, global_iter, noise_iter)
        feature = self.FE(input)
        z_rec = self.q(feature)
        z_rec = z_rec.view(B, -1) # B * z_dim
        return z_rec

    def forward_d(self, input):
        B, C, H, W = input.size()
        # input = self.add_guassian_noise(device, input, global_iter, noise_iter)
        feature = self.FE(input)
        out = self.d(feature)
        out = out.view(B, -1) #B * 1
        return out
    
    def forward(self, input):    
        B, C, H, W = input.size()
        # input = self.add_guassian_noise(device, input, global_iter, noise_iter)
        feature = self.FE(input)
        z_rec = self.q(feature).view(B, -1)
        out = self.d(feature).view(B, -1)
        return [z_rec, out]
    
class Weight_Seperated_Discriminator(BaseModel):

    def __init__(self, ndf, nc, z_dim, weight_init):
        super(Weight_Seperated_Discriminator, self).__init__()
        self.ndf = ndf
        self.nc = nc
        self.z_dim = z_dim

        self.FE = nn.Sequential()

        self.q = nn.Sequential(
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf, ndf*2, kernel_size= 4, stride = 2, padding=1, bias = False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*2, ndf*4, kernel_size = 4, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*4, ndf*8, kernel_size = 3, stride =1, padding =1, bias = False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*8, ndf*16, kernel_size=8, bias = False),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*16, ndf*16, kernel_size=1),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(ndf*16),

            nn.Conv2d(ndf*16, z_dim, kernel_size=1)
        )

        self.d = nn.Sequential(
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf, ndf*2, kernel_size= 4, stride = 2, padding=1, bias = False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*2, ndf*4, kernel_size = 4, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*4, ndf*8, kernel_size = 3, stride =1, padding =1, bias = False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*8, ndf*16, kernel_size=8, bias = False),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*16, 1, 1)
        )
        self.weight_init(weight_init)
    
    def forward_q(self,input):
        B, C, H, W = input.size()
        # new_input = self.add_guassian_noise(device, input, global_iter, noise_iter)
        feature = self.FE(input)
        z_rec = self.q(feature)
        z_rec = z_rec.view(B, -1) # B * z_dim
        return z_rec

    def forward_d(self, input):
        B, C, H, W = input.size()
        # input = self.add_guassian_noise(device, input, global_iter, noise_iter)
        feature = self.FE(input)
        out = self.d(feature)
        out = out.view(B, -1) #B * 1
        return out
    
    def forward(self, input):    
        B, C, H, W = input.size()
        # input = self.add_guassian_noise(device, input, global_iter, noise_iter)
        feature = self.FE(input)
        z_rec = self.q(feature).view(B, -1)
        out = self.d(feature).view(B, -1)
        return [z_rec, out]







