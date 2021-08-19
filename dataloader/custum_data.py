import os
import random
from torch.utils import data
from torchvision.datasets.folder import pil_loader
from PIL import Image

class CustumData(data.Dataset):
    def __init__(self, path_to_data =  "/home/hankyu/hankyu/disentangle/ibgan/data/dsprites", nc = 1, train = True, transforms = None):
        self.path_to_data = path_to_data
        self.train = train
        self.transforms = transforms
        self.nc = nc
        self.data = []
       
        self.prepare()
        print(len(self.data))
        
    def prepare(self):
        for img in os.listdir(self.path_to_data):
            self.data.append(os.path.join(self.path_to_data, img))
        

    def __getitem__(self, index) :

        data_path = self.data[index % len(self.data)]
        if self.nc == 3:
            data = pil_loader(data_path)
        else:
            data = Image.open(data_path).convert("L")

        if self.transforms is not None:
            data = self.transforms(data)
            
        return data

    def __len__(self):
        return len(self.data)      

