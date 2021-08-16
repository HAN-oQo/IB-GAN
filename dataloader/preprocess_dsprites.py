import torch
import os
from torchvision.utils import save_image
import numpy as np

data_path = '../data/dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
save_path = "../data/dsprites"
if not os.path.exists(save_path):
    os.makedirs(save_path)

data = np.load(root, encoding='latin1')
data = torch.from_numpy(data['imgs']).unsqueeze(1).float()

for i in range(data.size(0)):
    jpgfile = '{}.jpg'.format(i)
    save_image(data[i], os.path.join(save_path, jpgfile))

    print("Save {}th image into {}...".format(i, save_path))

