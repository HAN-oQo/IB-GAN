import torch
import torch.nn as nn

from models import BaseModel
from models.utils import *

class Generator(BaseModel):
    def __init__(self, ngf, nc, z_dim, r_dim):
        super(Generator, self).__init__()
        self.ngf= ngf
        self.nc = nc
        self.z_dim = z_dim
        self.r_dim = r_dim
        
        self.r = nn.Sequential(
            nn.Linear(z_dim, ngf*2),
            nn.BatchNorm1d(ngf*2),
            nn.ReLU(),
            nn.Linear(ngf*2, ngf),
            nn.BatchNorm1d(ngf),
            nn.ReLU(),
            nn.Linear(ngf, r_dim*2)
        )

        self.r_to_g = nn.Sequential(
            nn.Linear(r_dim, ngf*16),
            nn.BatchNorm1d(ngf*16),
            nn.ReLU(),

            nn.Linear(ngf*16, ngf*8*8*4),
            nn.BatchNorm1d(ngf*8*8*4),
            nn.ReLU()
        )

        self.g = nn.Sequential(
            nn.Conv2d(ngf*4, ngf*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(),

            nn.Conv2d(ngf*4, ngf*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(),
            
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size= 4, stride=2, padding= 1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(),

            nn.ConvTranspose2d(ngf*2, ngf, kernel_size= 4, stride=2, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(),

            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding =1),
            nn.Tanh()
        )

        self.weight_init()
    
    def forward_r(self, z):
        B, _ = z.size()
        out = self.r(z)
        r_mu, r_log_var = torch.split(out, [self.r_dim, self.r_dim], dim = -1)
        r = self.reparameterize(r_mu, r_log_var)
        return [r_mu, r_log_var, r]
    

    def forward_g(self, r):
        B, _ = r.size()
        input = self.r_to_g(r)
        input = input.view(B, -1, 8, 8)
        out = self.g(r)
        return out
    
    def forward(self, z):
        r_mu, r_log_var, r = self.forward_r(z)
        out = self.forward_g(r)
        return [r_mu, r_log_var, r, out]
    
    def traverse_latents(self, device, r, dim, start=-1.1, end=1.1, steps= 20):
        batch_size, _ = r.size()
        
        interpolation = torch.linspace(start= start, end =end, steps = steps)
        traversal_vectors = torch.zeros(batch_size*interpolation.size()[0], self.r_dim)
        for i in range(batch_size):
            r_base = r[i].clone()
            traversal_vectors[i*steps:(i+1)*steps, :] = r_base
            traversal_vectors[i*steps:(i+1)*steps, dim] = interpolation
        
        traversal_vectors = traversal_vectors.to(device)
        out = self.forward_g(traversal_vectors)
        # out_concat = torch.cat(out, dim=3)
        return out
