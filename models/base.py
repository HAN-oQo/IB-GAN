import torch
import torch.nn as nn
from models.utils import *
from abc import abstractmethod


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def weight_init(self, type):
        if type == "normal":
            for block in self._modules:
                try:
                    for m in self._modules[block]:
                        normal_init(m, 0.0, 0.02)
                except:
                    normal_init(block, 0.0, 0.02)
        elif type == "xavier":
            for block in self._modules:
                try:
                    for m in self._modules[block]:
                        xavier_init(m)
                except:
                    xavier_init(block)
        elif type == "kaiming":
            for block in self._modules:
                try:
                    for m in self._modules[block]:
                        kaiming_init(m)
                except:
                    kaiming_init(block)
        elif type == "orthogonal":
            for block in self._modules:
                try:
                    for m in self._modules[block]:
                        orthogonal_init(m)
                except:
                    orthogonal_init(block)
        else:
            raise(RuntimeError("Wrong weight init type"))


    def reparameterize(self, mu, log_var):
        
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + (eps * std)

    
    @abstractmethod
    def forward(self, input):
        pass
