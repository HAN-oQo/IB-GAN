import random
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms.transforms import CenterCrop
from dataloader.custum_data import *



def load_dataloader(dataset='dsprites', path_to_data= '/home/hankyu/hankyu/disentangle/ibgan/data/dsprites', train= True, nc =1 ,size= 64, batch_size = 64):
    
    if dataset == 'dsprites':
        
        if train:
            all_transforms = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5),
                                    std=(0.5))
            ])
        
        else:
            all_transforms = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5),
                                    std=(0.5))
            ])
        Data_Set = CustumData(path_to_data= path_to_data, nc= nc,train = train, transforms = all_transforms)
        print(len(Data_Set))
        data_loader = data.DataLoader(Data_Set, batch_size = batch_size, shuffle = True,  num_workers = 3, drop_last= True)
        return data_loader, len(Data_Set)

    elif dataset == '3dchairs':
        all_transforms = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5),
                                std=(0.5))
            
        ])
        Data_Set = CustumData(path_to_data= path_to_data,nc=nc, train = train, transforms = all_transforms)
        print(len(Data_Set))
        data_loader = data.DataLoader(Data_Set, batch_size = batch_size, shuffle = True,  num_workers = 20, drop_last= True)

        return data_loader, len(Data_Set)

    elif dataset == 'celeba':

        all_transforms = transforms.Compose([
            transforms.CenterCrop(148),
            transforms.Resize([size, size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
            
        ])
        Data_Set = CustumData(path_to_data= path_to_data, nc= nc, train = train, transforms = all_transforms)
        print(len(Data_Set))
        data_loader = data.DataLoader(Data_Set, batch_size = batch_size, shuffle = True,  num_workers = 8, drop_last= True)

        return data_loader, len(Data_Set)

    else:
        raise(RuntimeError("Wrong Dataset"))

